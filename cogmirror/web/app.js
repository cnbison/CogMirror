/* CogMirror 学生端单页（无框架纯 JS，形态对齐 ECOS web/student）.
 *
 * 视图流转：welcome -> quiz（逐题：自评 -> 作答 -> 判分 -> [答错追问解释]
 * -> 提交引擎）-> map（认知地图 + 建议练习循环）-> welcome。
 * 数据接口见 cogmirror/webui.py（grade 纯判分 / commit 更新落库 两段式，
 * 镜像 CLI 流程）。
 */
"use strict";

const TOPICS = [
  ["", "全部 topic"],
  ["python.variables", "变量赋值"],
  ["python.loops", "循环"],
  ["python.functions", "函数"],
  ["python.recursion", "递归"],
  ["python.scope", "作用域"],
];
const LEVELS = [["", "不限层级"], ["L1", "L1 记忆"], ["L2", "L2 理解"], ["L3", "L3 应用"],
  ["L4", "L4 分析"], ["L5", "L5 评价"], ["L6", "L6 创造"]];

const app = document.getElementById("app");
const S = { user: "local_user", questions: [], idx: 0, graded: null, selfConf: null };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, body) {
  const opt = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opt);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `请求失败（${res.status}）`);
  return data;
}

function qs(name) { return new URLSearchParams(location.search).get(name); }

/* ── 欢迎视图 ─────────────────────────────────────────────────── */

async function showWelcome() {
  const init = await api(`/api/init?user=${encodeURIComponent(S.user)}`);
  document.getElementById("userTag").textContent = `用户：${S.user}`;
  const info = [];
  if (init.overview) info.push(`<div class="kv"><b>进度概览</b><span>${esc(init.overview)}</span></div>`);
  if (init.struggles && init.struggles.length) {
    info.push(`<div class="kv"><b>上次卡住</b><span>${esc(init.struggles.join("、"))}</span></div>`);
  }
  app.innerHTML = `
  <div class="card">
    <h2>${init.is_new ? "开始你的第一组题" : `欢迎回来`}</h2>
    ${init.is_new ? `<p class="muted">每道题先自评把握再作答；答错后可以写一句「为什么这么答」，
      系统会从中识别典型的概念误解。答完会画出你的认知地图。</p>`
      : `<p class="muted">已完成 ${init.n_responses} 次作答。</p>`}
    ${info.join("")}
    <div class="settings">
      <label>题数 <input id="setN" type="number" min="1" max="51" value="10"></label>
      <label>topic <select id="setTopic">${TOPICS.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select></label>
      <label>层级 <select id="setLevel">${LEVELS.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select></label>
      <label class="chk"><input id="setReview" type="checkbox"> 只练错题</label>
    </div>
    <div class="row">
      ${init.quiz_in_progress ? `<button class="primary" id="btnResume">
        继续答题（还剩 ${init.quiz_in_progress.remaining} 题）</button>` : ""}
      <button class="${init.quiz_in_progress ? "" : "primary"}" id="btnStart">开始答题</button>
      ${init.is_new ? "" : `<button id="btnMap">只看认知地图</button>`}
    </div>
  </div>`;
  const btnResume = document.getElementById("btnResume");
  if (btnResume) btnResume.onclick = async () => {
    const data = await api(`/api/quiz/resume?user=${encodeURIComponent(S.user)}`);
    S.questions = data.questions;
    S.idx = 0;
    if (S.questions.length) renderQuestion();
  };
  document.getElementById("btnStart").onclick = () => {
    const review = document.getElementById("setReview").checked;
    const n = review ? 0 : Number(document.getElementById("setN").value) || 10;
    const topic = document.getElementById("setTopic").value;
    const level = document.getElementById("setLevel").value;
    // fresh=1：新题优先，跳过已作答过的题（不够时已答题补尾）
    const q = `?user=${encodeURIComponent(S.user)}&n=${n}&topic=${encodeURIComponent(topic)}&level=${encodeURIComponent(level)}${review ? "&review=1" : "&fresh=1"}`;
    startQuiz(q);
  };
  const btnMap = document.getElementById("btnMap");
  if (btnMap) btnMap.onclick = showMap;
}

/* ── 答题视图 ─────────────────────────────────────────────────── */

async function startQuiz(query) {
  const data = await api(`/api/quiz${query}`);
  S.questions = data.questions;
  S.idx = 0;
  if (!S.questions.length) {
    app.innerHTML = `<div class="card"><p>当前筛选条件下没有题目。</p>
      <button onclick="showMap()">看认知地图</button></div>`;
    return;
  }
  renderQuestion();
}

function renderQuestion(feedback) {
  const q = S.questions[S.idx];
  const n = S.questions.length;
  let answerHtml;
  if (q.qtype === "choice") {
    answerHtml = `<div class="options">${q.options.map((opt, i) => `
      <label class="option"><input type="radio" name="opt" value="${i}">
      <span><b>${i}.</b> ${esc(opt)}</span></label>`).join("")}</div>`;
  } else if (q.qtype === "fill") {
    answerHtml = `<input class="fill" id="answer" placeholder="输入你的答案">`;
  } else {
    answerHtml = `<p class="muted small">写代码：随时可以修改任意一行，写完点提交。</p>
      <textarea class="code" id="answer" rows="10" spellcheck="false"
        placeholder="def my_func(...):"></textarea>`;
  }
  app.innerHTML = `
  <div class="card quiz">
    <div class="qhead"><span class="badge">题 ${S.idx + 1}/${n}</span>
      <span class="meta">${esc(q.topic_label)} / ${esc(q.bloom_level)}</span></div>
    <div class="prompt">${esc(q.prompt)}</div>
    ${feedback ? `<div class="live">${feedback.map((l) => `<p>${esc(l)}</p>`).join("")}</div>` : ""}
    <div class="selfconf">
      <label>答题前自评：多大把握答对？（0-100，留空跳过）
        <input id="selfconf" type="number" min="0" max="100" placeholder="如 80"></label>
    </div>
    ${answerHtml}
    <div class="row"><button class="primary" id="btnSubmit">提交</button></div>
    <div id="result"></div>
  </div>`;
  document.getElementById("btnSubmit").onclick = submitAnswer;
  const input = document.getElementById("answer");
  if (input && q.qtype !== "choice") input.focus();
}

async function submitAnswer() {
  const q = S.questions[S.idx];
  const scRaw = document.getElementById("selfconf").value.trim();
  S.selfConf = scRaw === "" ? null : Math.min(100, Math.max(0, Number(scRaw))) / 100;
  let answer;
  if (q.qtype === "choice") {
    const checked = document.querySelector('input[name="opt"]:checked');
    if (!checked) { alert("请先选择一个选项"); return; }
    answer = checked.value;
  } else {
    answer = document.getElementById("answer").value;
  }
  const btn = document.getElementById("btnSubmit");
  btn.disabled = true;
  btn.textContent = "判分中…";
  try {
    const graded = await api("/api/grade", { user: S.user, problem_id: q.problem_id, answer });
    if (graded.syntax_error) {
      // 语法错误 = 笔误不是概念信号：保留编辑器就地修正重交（不进判分
      // 结果视图、不落库）；「放弃此题」按 0 分提交作保底
      S.graded = graded;
      showSyntaxBanner(graded.syntax_error);
      btn.disabled = false;
      btn.textContent = "提交";
      document.getElementById("answer").focus();
      return;
    }
    renderGraded(graded);
  } catch (e) {
    alert(e.message);
    btn.disabled = false;
    btn.textContent = "提交";
  }
}

function showSyntaxBanner(err) {
  document.getElementById("result").innerHTML = `
    <div class="syntax-err">
      <p><b>${esc(err.message)}</b></p>
      ${err.line ? `<pre class="case">${esc(err.line)}</pre>` : ""}
      <p class="muted small">语法错误通常是笔误，不算你对概念的掌握——修正上面的代码后重新提交即可；
        确实修不出来就放弃本题（计 0 分）。</p>
      <button id="btnGiveup">放弃此题（计 0 分）</button>
    </div>`;
  document.getElementById("btnGiveup").onclick = () => renderGraded(S.graded);
  document.getElementById("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderGraded(graded) {
  S.graded = graded;
  const q = S.questions[S.idx];
  const scoreTxt = (graded.score * 100).toFixed(0) + "%" +
    (graded.score > 0 && graded.score < 1 ? "（部分正确）" : "");
  const details = (graded.details || []).slice(0, 8).map((d) => d.error
    ? `<p class="err">${esc(d.error)}</p>`
    : `<p class="case">用例 ${esc(d.args)}：期望 ${esc(d.expected)}，得到 ${esc(d.got)}
       ${d.passed ? "✓" : "✗"}</p>`).join("");
  const explain = graded.correct ? "" : `
    <div class="explain">
      <label>为什么这么答？用一句话说说你的理由（可直接跳过，用于识别典型误解）
        <textarea id="explain" rows="2"></textarea></label>
    </div>`;
  const ex = graded.option_explanation || graded.key_point || "";
  document.getElementById("result").innerHTML = `
    <div class="score ${graded.correct ? "ok" : "bad"}">得分：${scoreTxt}</div>
    ${details}${ex ? `<p class="keypoint">${esc(ex)}</p>` : ""}
    ${explain}
    <div class="row"><button class="primary" id="btnNext">继续</button></div>`;
  document.getElementById("btnNext").onclick = commitAnswer;
  document.getElementById("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function commitAnswer() {
  const explanationEl = document.getElementById("explain");
  const explanation = explanationEl ? explanationEl.value.trim() : "";
  const data = await api("/api/commit", {
    user: S.user, explanation,
    self_confidence: S.selfConf === null ? null : Number(S.selfConf.toFixed(2)),
  });
  const feedback = data.live_feedback || [];
  S.idx += 1;
  if (S.idx >= S.questions.length) {
    await showMap(true, feedback);
  } else {
    renderQuestion(feedback);
  }
}

/* ── 认知地图视图 ─────────────────────────────────────────────── */

function bar(value) {
  const pct = (value * 100).toFixed(0);
  const cls = value >= 0.8 ? "b-good" : value >= 0.6 ? "b-mid" : "b-weak";
  return `<div class="bar"><div class="${cls}" style="width:${pct}%"></div><span>${pct}%</span></div>`;
}

async function showMap(fromQuiz, pendingFeedback) {
  const data = fromQuiz
    ? await api("/api/quiz/finish", { user: S.user })
    : await api(`/api/map?user=${encodeURIComponent(S.user)}`);
  renderMap(data, pendingFeedback || []);
}

function renderMap(d, feedback) {
  const secs = [];
  secs.push(section("整体解读", `<p>${esc(d.interpretation)}</p>`));

  const dims = (d.dims || []).map((x) => `
    <div class="dimrow"><span class="dimname">${x.dim} ${esc(x.label)}</span>
      ${x.mastery === null ? `<span class="muted">${esc(x.note)}</span>` : bar(x.mastery)}</div>`).join("");
  let dimSec = `<div class="dims">${dims}</div>`;
  if (d.calibration) dimSec += `<p class="muted small">${esc(d.calibration)}</p>`;
  secs.push(section("5 维状态（掌握概率）", dimSec));

  if (d.delta_lines && d.delta_lines.length) {
    secs.push(section("与上次相比", d.delta_lines.map((l) => `<p>${esc(l)}</p>`).join("")));
  }

  const bloom = (d.bloom || []).map((b) => `
    <div class="dimrow"><span class="dimname">${esc(b.label)}</span>
      ${b.covered ? bar(b.value) : `<span class="muted">（暂未测量）</span>`}</div>`).join("")
    + `<p class="muted small">当前主导层级：${esc(d.bloom_dominant || "暂未测量")}</p>`;
  secs.push(section("Bloom 六层分布（各层掌握概率）", bloom));

  if (d.illusory_hits && d.illusory_hits.length) {
    const rows = d.illusory_hits.map((h) => `<p>题 ${esc(h.problem_id)}：
      自评 ${(h.self_confidence * 100).toFixed(0)}%，实际得分 ${(h.score * 100).toFixed(0)}%
      （落差 ${(h.gap * 100).toFixed(0)}%）</p>`).join("")
      + `<p class="muted small">这些地方『感觉会』可能掩盖了『其实还没会』，建议重做并讲出理由。</p>`;
    secs.push(section("伪自信点", rows));
  }

  if (d.misc_hits && d.misc_hits.length) {
    const rows = d.misc_hits.map((h) => `<p>题 ${esc(h.problem_id)}：${esc(h.name)}
      （置信度 ${(h.confidence * 100).toFixed(0)}%）「${esc(h.evidence_text)}」</p>`).join("");
    secs.push(section("误解点", rows));
  } else {
    secs.push(section("误解点", `<p class="muted">本次无（答错后的「为什么这么答」追问是检测输入，跳过则无数据）</p>`));
  }

  const tc = d.tc || { liminal: [], crossed: [] };
  let tcHtml = "";
  if (tc.liminal.length) {
    tcHtml += `<p>正在跨越中（这不是退步，是学习的正常中间态）：</p>` +
      tc.liminal.map((t) => `<p>「${esc(t.name)}」（跨越进度 ${(t.progress * 100).toFixed(0)}%，${esc(t.remaining)}）</p>`).join("");
  }
  if (tc.crossed.length) {
    tcHtml += `<p>已跨越（恭喜，这些概念你已经真正掌握）：</p>` +
      tc.crossed.map((t) => `<p>「${esc(t)}」</p>`).join("");
  }
  if (!tcHtml) tcHtml = `<p class="muted">当前无正在跨越中的概念</p>`;
  secs.push(section("临界概念", tcHtml));

  if (d.retest && d.retest.length) {
    secs.push(section("复习提示（曾掌握、正在遗忘）", d.retest.map((r) =>
      `<p>「${esc(r.skill)}」${r.days} 天未练，掌握概率从 ${(r.peak * 100).toFixed(0)}%
       掉到 ${(r.decayed * 100).toFixed(0)}%</p>`).join("")));
  }

  if (d.trend) {
    secs.push(section(`近几次趋势（${d.trend.n_sessions} 次会话末对比）`, `<p>${esc(d.trend.line)}</p>`));
  }

  let practiceHtml = `<p>${esc(d.suggestion)}</p>`;
  if (d.practice_command) practiceHtml += `<p class="muted small">命令行直达：${esc(d.practice_command)}</p>`;
  const sp = d.suggested_practice;
  if (sp) {
    const kind = sp.level ? "的 L3 题" : (d.retest && d.retest.some((r) => r.skill === sp.topic_label) ? "的复测题" : "的基础题");
    practiceHtml += `<div class="row"><button class="primary" id="btnPractice">
      按建议现在练 ${sp.n} 道「${esc(sp.topic_label)}」${kind}</button></div>`;
  }
  practiceHtml += `<div class="row"><button id="btnBack">返回</button></div>`;
  secs.push(section("一句话建议", practiceHtml));

  app.innerHTML =
    (feedback.length ? `<div class="card live">${feedback.map((l) => `<p>${esc(l)}</p>`).join("")}</div>` : "")
    + `<div class="card map"><h2>你的认知地图</h2>
       <p class="muted small">怎么看：每行条形 = 该维度的掌握概率，越接近 100% 越稳；作答越多，数值越准。</p>
       ${secs.join("")}</div>`;
  const btnP = document.getElementById("btnPractice");
  if (btnP) btnP.onclick = () => startQuiz(
    `?user=${encodeURIComponent(S.user)}&n=${sp.n}&topic=${encodeURIComponent(sp.topic)}&level=${encodeURIComponent(sp.level || "")}`);
  document.getElementById("btnBack").onclick = showWelcome;
}

function section(title, inner) {
  return `<section><h3>${esc(title)}</h3>${inner}</section>`;
}

/* ── 启动 ─────────────────────────────────────────────────────── */

S.user = qs("user") || "local_user";
showWelcome().catch((e) => {
  app.innerHTML = `<div class="card"><p class="err">加载失败：${esc(e.message)}</p></div>`;
});
