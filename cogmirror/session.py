"""纵向档案聚合（方案 P5 B1，纯函数，零 LLM）.

借 PersonalAGI 主动召回/4 层记忆的「相关历史主动浮现」模式（L1/L2 的
效果），实现退化为会话开始时从已有 SQLite 聚合：上次卡点 + 跨会话趋势。
不引入 Qdrant/向量（SOMEDAY：学习者数量进入多用户阶段）。

会话边界（无 session 边界表，同 P2 免维护理由）：按快照时间聚类--
相邻 belief_snapshots 间隔超过 SESSION_GAP_MINUTES 视为新会话。每次
run_session 末 save_state 一条，一次 CLI 运行多轮练习会产生多条快照但
间隔短，聚进同一会话。可从 DB 幂等重算。

趋势是快照序列的派生视图，不落库；显示层（CLI）负责"数据不足"诚实
标注，本模块只返回数据与 None/空语义。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# 相邻快照间隔超过该分钟数 -> 视为新会话（一次 CLI 运行内的练习轮间隔
# 为秒-分钟级，跨次运行通常远超；阈值是工程折中，非认知参数）
SESSION_GAP_MINUTES = 30

_TREND_DIMS = ("K", "P", "S")


def _parse_iso(raw: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return None


def _snapshot_clusters(snapshots: List[dict]) -> List[List[dict]]:
    """把升序快照按时间间隔切成会话聚类."""
    clusters: List[List[dict]] = []
    prev_time: Optional[datetime] = None
    for s in snapshots:
        t = _parse_iso(s.get("created_at", ""))
        if prev_time is not None and t is not None:
            gap_minutes = (t - prev_time).total_seconds() / 60.0
            if gap_minutes > SESSION_GAP_MINUTES:
                clusters.append([])
        if not clusters:
            clusters.append([])
        clusters[-1].append(s)
        if t is not None:
            prev_time = t
    return clusters


def last_session_struggles(db, user_id: str) -> List[str]:
    """上次会话答错/部分正确的 skill（去重，按首次出现顺序）.

    会话 = 快照聚类；"上次"取最后一个聚类（调用时机为会话开始的欢迎行，
    本次运行尚未产生新快照）。窗口 = (上一聚类末快照, 本聚类末快照 + 半个
    gap]：实际时序是先作答后存快照（每轮 run_session 末 save_state），
    会话内作答发生在首快照之前，故起点放宽到上一聚类末；终点加半 gap
    容纳末快照后紧邻的零星作答。无快照（首次运行）返回空。
    """
    snapshots = db.load_snapshots(user_id)
    if not snapshots:
        return []
    clusters = [c for c in _snapshot_clusters(snapshots) if c]
    last = clusters[-1]
    start: Optional[datetime] = None
    if len(clusters) >= 2:
        start = _parse_iso(clusters[-2][-1]["created_at"])
    end = _parse_iso(last[-1]["created_at"])
    if end is None:
        return []
    end = end + timedelta(minutes=SESSION_GAP_MINUTES / 2)
    struggles: List[str] = []
    for r in db.load_responses(user_id):
        t = _parse_iso(r.get("created_at", ""))
        if t is None:
            continue
        if start is not None and t <= start:
            continue
        if t > end:
            continue
        skill = r.get("skill_id")
        if skill and (r.get("score") or 0.0) < 0.6 and skill not in struggles:
            struggles.append(skill)
    return struggles


def multi_session_trend(db, user_id: str, n: int = 3) -> Dict[str, Tuple[float, float, int]]:
    """最近 n 个会话末的 K/P/S 掌握概率趋势.

    返回 {dim: (first, last, n_sessions)}：取最近 n 个会话聚类各自的末条
    快照，解析 state_json 的 mastery_prob 组成序列，first = 首个会话末值、
    last = 末个会话末值。会话数 < 2 时返回空 dict（无法谈趋势）。
    """
    snapshots = db.load_snapshots(user_id)
    if not snapshots:
        return {}
    clusters = [c for c in _snapshot_clusters(snapshots) if c]
    if len(clusters) < 2:
        return {}
    tails = clusters[-n:]
    series: Dict[str, List[float]] = {dim: [] for dim in _TREND_DIMS}
    for cluster in tails:
        state = json.loads(cluster[-1]["state_json"])
        for dim in _TREND_DIMS:
            series[dim].append(float(state.get(dim, {}).get("mastery_prob", 0.0)))
    return {
        dim: (values[0], values[-1], len(tails))
        for dim, values in series.items()
    }


def trend_line(trend: Dict[str, Tuple[float, float, int]]) -> str:
    """[近几次趋势] 单行文案（K/P/S 各一短句）；空 trend 返回空串.

    与 CLI 的显示细节耦合，放在这里让 CLI 只做注入（显示层克制：一行，
    维度级首末值可回溯到具体快照证据）。
    """
    if not trend:
        return ""
    names = {"K": "知识", "P": "程序技能", "S": "策略"}
    parts = []
    for dim in _TREND_DIMS:
        if dim not in trend:
            continue
        first, last, n_sess = trend[dim]
        d = last - first
        if abs(d) < 0.02:
            continue
        sign = "+" if d > 0 else ""
        parts.append(f"{names[dim]} {sign}{d:.0%}（{n_sess} 次会话 "
                     f"{first:.0%} -> {last:.0%}）")
    if not parts:
        return "近几次会话 K/P/S 维度基本稳定，无显著趋势。"
    return "；".join(parts)
