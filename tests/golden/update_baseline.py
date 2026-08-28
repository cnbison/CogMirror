"""重建黄金基线：python -m tests.golden.update_baseline

纪律（docs/implementation-plan.md 2.6）：基线更新必须带文档化 diff（改了什么、
为什么），不允许静默覆盖。本脚本打印每个 case 与旧基线的差异摘要，人工核对
后再提交。
"""

from __future__ import annotations

from .runner import compare_summary, load_baseline, run_all, write_baseline
from .sequences import SEQUENCES


def main() -> None:
    old = load_baseline()
    cases = {}
    print(f"共 {len(SEQUENCES)} 条黄金序列")
    for seq, result in run_all():
        cases[seq["name"]] = {
            "passed": not result.expect_failures,
            "summary": result.summary,
        }
        if result.expect_failures:
            print(f"[expect FAIL] {seq['name']}:")
            for f in result.expect_failures:
                print(f"    {f}")

    baseline = {
        "meta": {
            "note": "黄金基线：引擎在 tests/golden/sequences.py 上的数值快照。"
                    "更新须带文档化 diff（方案 2.6），对比容差见 runner.TOL。",
        },
        "cases": cases,
    }
    if old:
        print("\n与旧基线的差异（须在提交说明中逐一解释）：")
        any_diff = False
        for name, entry in cases.items():
            old_entry = old.get("cases", {}).get(name)
            if old_entry is None:
                print(f"  {name}: 新增 case（冷启动，不报回归）")
                any_diff = True
                continue
            drifts = compare_summary(entry["summary"], old_entry["summary"])
            passed_flip = entry["passed"] != old_entry["passed"]
            if drifts or passed_flip:
                any_diff = True
                print(f"  {name}: passed {old_entry['passed']} -> {entry['passed']}")
                for d in drifts:
                    print(f"    {d}")
        for name in old.get("cases", {}):
            if name not in cases:
                any_diff = True
                print(f"  {name}: 序列已删除")
        if not any_diff:
            print("  （无差异）")
    else:
        print("\n无旧基线，全新生成。")

    write_baseline(baseline)
    print("\n已写入 tests/golden/baseline.json")


if __name__ == "__main__":
    main()
