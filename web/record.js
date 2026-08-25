// 錄影用的離線畫面產生器。
//
// 整個 1920x1080 畫面都畫在同一張 canvas 上，不用任何 DOM 排版，
// 這樣每一格 frame 都完全可重現。九連環又比打磚塊更徹底——它連亂數都幾乎沒有，
// 除了「亂走」那一段之外，同一份權重每次錄出來是一模一樣的影片。
//
// puppeteer 那邊的用法：
//   const plan = await window.__rec.init();
//   for (let i = 0; i < plan.totalFrames; i++) { window.__rec.renderFrame(); screenshot(); }

import { RingsEngine, legalMask, distance, optimalAction } from "./engine.js";
import { Policy } from "./nn.js";

const FPS = 30;
const W = 1920, H = 1080;
const GAP_SEC = 0.9;      // 每段旁白唸完後留白
const HOLD_SEC = 1.3;     // 每局結束後停在終盤畫面

const C = {
  bg: "#060911", panel: "#0f172a", line: "#1e293b",
  fg: "#e2e8f0", dim: "#94a3b8", faint: "#64748b",
  accent: "#38bdf8", warn: "#fbbf24", good: "#4ade80", bad: "#f87171",
};
const FONT = '"Microsoft JhengHei", "Noto Sans TC", system-ui, sans-serif';
const MONO = 'ui-monospace, "Cascadia Mono", Consolas, monospace';

const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d", { alpha: false });

// 左半邊：謎題本體 + 距離曲線。右半邊：文字與 Q 值。
const LX = 78, LW = 1030;
const PUZ_TOP = 150, BAR_Y = PUZ_TOP + 150, RING_R = 40, RING_GAP = 108, OFF_DY = 104;
const CURVE = { x: LX, y: 620, w: LW, h: 300 };
const RX = LX + LW + 82;
const RW = W - RX - LX;

let cfg, policy, bench, ablation, timing, script, plan;
let frameIndex = 0;
let cur = null;

// ---------------------------------------------------------------- 初始化

async function init() {
  cfg = await (await fetch("../shared/config.json", { cache: "no-store" })).json();
  policy = new Policy(await (await fetch("policy.json", { cache: "no-store" })).json());
  bench = await (await fetch("../ml/checkpoints/benchmark.json", { cache: "no-store" })).json();
  ablation = await (await fetch("../ml/checkpoints/ablation.json", { cache: "no-store" })).json();
  timing = await (await fetch("../out/voice/timing.json", { cache: "no-store" })).json();
  script = await (await fetch("../tools/script.json", { cache: "no-store" })).json();

  const byId = Object.fromEntries(timing.map((t) => [t.id, t.duration]));

  let start = 0;
  plan = { fps: FPS, width: W, height: H, scenes: [] };
  for (const sc of script.scenes) {
    const frames = Math.round((byId[sc.id] + GAP_SEC) * FPS);
    const entry = {
      id: sc.id, kind: sc.kind, mode: sc.mode || null, label: sc.label || "",
      start, frames, seed: sc.seed ?? null,
    };
    if (sc.kind === "puzzle" || sc.kind === "title") {
      // 先把這局跑完，才知道總共幾步，才能算「一格 frame 要推進幾步」，
      // 讓謎題剛好在旁白唸完前解完（或撞到步數上限）。
      const r = simulate(sc.mode, sc.seed ?? 0, sc.maxPlaySteps || 0);
      entry.totalSteps = r.steps;
      entry.solved = r.solved;
      entry.maxPlaySteps = sc.maxPlaySteps || 0;
      const holdFrames = Math.round(HOLD_SEC * FPS);
      entry.playFrames = Math.max(1, frames - holdFrames);
      entry.stepsPerFrame = entry.totalSteps / entry.playFrames;
    }
    plan.scenes.push(entry);
    start += frames;
  }
  plan.totalFrames = start;
  plan.durationSec = +(start / FPS).toFixed(2);
  frameIndex = 0;
  cur = null;
  return plan;
}

// ---------------------------------------------------------------- 四種 policy

function actFor(mode, e, rand) {
  const m = e.legal();
  if (mode === "optimal") return optimalAction(e.s);
  if (mode === "dqn") return policy.act(e.getObservation(), m).action;
  if (mode === "greedy") {
    // 直覺策略：能拿下來就拿下來，逼不得已才裝回去。九連環一定要退才能進，所以它會卡死。
    for (let i = 0; i < e.n; i++) if (m[i] && e.s[i] === 1) return i;
    for (let i = 0; i < e.n; i++) if (m[i]) return i;
  }
  // random
  const idx = [];
  for (let i = 0; i < m.length; i++) if (m[i]) idx.push(i);
  return idx[Math.floor(rand() * idx.length) % idx.length];
}

// 小型可重現亂數。「亂走」那一段必須用它，不然每次錄出來的路徑都不一樣。
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function simulate(mode, seed, maxPlaySteps) {
  const e = new RingsEngine(cfg);
  const rand = mulberry32(seed);
  const cap = maxPlaySteps || Infinity;
  while (!e.over && e.steps < cap) e.step(actFor(mode, e, rand));
  return { steps: e.steps, solved: e.solved };
}

// ---------------------------------------------------------------- 每格 frame

function renderFrame() {
  const sc = plan.scenes.find((s) => frameIndex >= s.start && frameIndex < s.start + s.frames)
          || plan.scenes[plan.scenes.length - 1];
  const local = frameIndex - sc.start;

  if (!cur || cur.id !== sc.id) {
    cur = { id: sc.id, sc, acc: 0, trace: [], engine: null, rand: mulberry32(sc.seed ?? 0) };
    if (sc.kind === "puzzle" || sc.kind === "title") {
      cur.engine = new RingsEngine(cfg);
      cur.trace.push([0, cur.engine.dist]);
      if (sc.mode === "dqn") cur.lastQ = policy.qValues(cur.engine.getObservation());
    }
  }

  if ((sc.kind === "puzzle" || sc.kind === "title") && local < sc.playFrames) {
    cur.acc += sc.stepsPerFrame;
    const n = Math.floor(cur.acc);
    cur.acc -= n;
    const cap = sc.maxPlaySteps || Infinity;
    for (let k = 0; k < n && !cur.engine.over && cur.engine.steps < cap; k++) {
      const a = actFor(sc.mode, cur.engine, cur.rand);
      if (sc.mode === "dqn") cur.lastQ = policy.qValues(cur.engine.getObservation());
      cur.lastA = a;
      cur.engine.step(a);
      cur.trace.push([cur.engine.steps, cur.engine.dist]);
    }
  }

  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  if (sc.kind === "shaping") drawShaping(local);
  else if (sc.kind === "results") drawResults(local);
  else {
    drawHeader();
    drawPuzzle(sc);
    drawCurve(sc);
    drawSide(sc, local);
  }
  drawProgress();
  frameIndex++;
  return frameIndex;
}

// ---------------------------------------------------------------- 版面

function drawHeader() {
  ctx.textAlign = "left";
  ctx.fillStyle = C.fg;
  ctx.font = `700 40px ${FONT}`;
  ctx.fillText("九連環 × DQN", LX, 74);
  ctx.fillStyle = C.faint;
  ctx.font = `20px ${FONT}`;
  ctx.fillText("自己寫謎題 · 自己寫環境 · 自己訓練 agent", LX + 290, 74);
}

function drawPuzzle(sc) {
  const e = cur.engine;
  ctx.fillStyle = "#0b1020";
  roundRect(ctx, LX, PUZ_TOP - 40, LW, 400, 12);
  ctx.fill();
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  ctx.stroke();
  if (!e) return;

  const mask = e.legal();
  const xOf = (i) => LX + LW - 96 - i * RING_GAP;
  const baseY = BAR_Y + 172;

  // 底板
  ctx.fillStyle = "#1e293b";
  roundRect(ctx, LX + 40, baseY, LW - 80, 14, 5);
  ctx.fill();

  // 橫桿（劍），左端是握把
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 11;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(LX + 106, BAR_Y);
  ctx.lineTo(LX + LW - 46, BAR_Y);
  ctx.stroke();
  ctx.lineWidth = 8;
  ctx.beginPath();
  ctx.ellipse(LX + 84, BAR_Y, 26, 38, 0, 0, Math.PI * 2);
  ctx.stroke();

  for (let i = 0; i < e.n; i++) {
    const cx = xOf(i);
    const on = e.s[i] === 1;
    const cy = on ? BAR_Y : BAR_Y - OFF_DY;
    const legal = mask[i] === 1;

    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(cx, baseY);
    ctx.lineTo(cx, cy + RING_R);
    ctx.stroke();

    const isLast = i === cur.lastA;
    ctx.strokeStyle = isLast ? "#fde047" : legal ? C.accent : "#475569";
    ctx.lineWidth = 11;
    ctx.beginPath();
    ctx.arc(cx, cy, RING_R, 0, Math.PI * 2);
    ctx.stroke();

    if (legal) {
      ctx.fillStyle = "rgba(56,189,248,0.9)";
      ctx.font = `20px ${FONT}`;
      ctx.textAlign = "center";
      ctx.fillText(on ? "▲" : "▼", cx, cy - RING_R - 14);
    }
    ctx.fillStyle = legal ? "#cbd5e1" : "#475569";
    ctx.font = `20px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText(String(i + 1), cx, baseY + 40);
  }
  ctx.textAlign = "left";

  ctx.fillStyle = C.faint;
  ctx.font = `22px ${MONO}`;
  ctx.fillText(`state ${Array.from(e.s).join("")}`, LX + 28, PUZ_TOP - 4);
  ctx.textAlign = "right";
  ctx.fillStyle = e.dist === 0 ? C.good : C.dim;
  ctx.fillText(e.dist === 0 ? "解開了" : `離終點 ${e.dist} 步`, LX + LW - 28, PUZ_TOP - 4);
  ctx.textAlign = "left";
}

// 這支影片真正的主角：距離對步數的曲線。
// 最優解是一條斜直線；貪心會卡在半空中；亂走會醉步。看曲線比看環清楚得多。
function drawCurve(sc) {
  const { x, y, w, h } = CURVE;
  ctx.fillStyle = "#0b1020";
  roundRect(ctx, x, y, w, h, 12);
  ctx.fill();
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  ctx.stroke();

  const D0 = cur.engine ? cur.engine.startDistance : 341;
  const maxX = Math.max(D0, sc.totalSteps || D0);
  const pad = { l: 78, r: 26, t: 44, b: 40 };
  const X = (v) => x + pad.l + (v / maxX) * (w - pad.l - pad.r);
  const Y = (v) => y + h - pad.b - (v / D0) * (h - pad.t - pad.b);

  ctx.fillStyle = C.faint;
  ctx.font = `20px ${FONT}`;
  ctx.fillText("離終點還有幾步（縱軸）／已經動了幾步（橫軸）", x + 24, y + 30);

  ctx.strokeStyle = "#16213a";
  ctx.font = `16px ${MONO}`;
  for (let i = 0; i <= 4; i++) {
    const v = (D0 / 4) * i, yy = Y(v);
    ctx.beginPath();
    ctx.moveTo(x + pad.l, yy);
    ctx.lineTo(x + w - pad.r, yy);
    ctx.stroke();
    ctx.fillStyle = C.faint;
    ctx.textAlign = "right";
    ctx.fillText(v.toFixed(0), x + pad.l - 12, yy + 6);
  }
  ctx.textAlign = "left";

  // 最優解的參考線：341 步走完，一步都不多
  ctx.strokeStyle = "rgba(148,163,184,0.5)";
  ctx.setLineDash([8, 8]);
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(X(0), Y(D0));
  ctx.lineTo(X(D0), Y(0));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(148,163,184,0.8)";
  ctx.font = `17px ${FONT}`;
  ctx.fillText(`最優解 ${D0} 步`, X(D0 * 0.52) + 10, Y(D0 * 0.52) - 12);

  const t = cur.trace;
  if (t.length > 1) {
    ctx.strokeStyle = sc.mode === "dqn" ? C.accent : sc.mode === "optimal" ? C.good : C.warn;
    ctx.lineWidth = 3.5;
    ctx.beginPath();
    t.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
    ctx.stroke();
    const last = t[t.length - 1];
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.arc(X(last[0]), Y(last[1]), 6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawSide(sc, local) {
  let y = PUZ_TOP - 26;

  if (sc.kind === "title") {
    ctx.fillStyle = C.accent;
    ctx.font = `700 60px ${FONT}`;
    ctx.fillText("341 步，", RX, y + 58);
    ctx.fillText("一步都不能多", RX, y + 134);
    y += 206;
    ctx.fillStyle = C.dim;
    ctx.font = `25px ${FONT}`;
    ["九連環有精確的最優解。", "所以這一次，我們知道 AI 到底差了幾步。"]
      .forEach((l, i) => ctx.fillText(l, RX, y + i * 40));
    y += 108;
    for (const [k, v] of [
      ["謎題", "九連環，2⁹ = 512 個狀態"],
      ["最優解", "341 步（格雷碼算得出來）"],
      ["演算法", "Double DQN + Dueling + 5-step"],
      ["狀態", "9 個 bit + 9 個合法遮罩"],
      ["訓練量", "800,000 個 transition，約 3 分鐘"],
      ["評估", "窮舉 511 個起點，沒有取樣雜訊"],
    ]) {
      ctx.fillStyle = C.faint;
      ctx.font = `22px ${FONT}`;
      ctx.fillText(k, RX, y);
      ctx.fillStyle = C.fg;
      ctx.fillText(v, RX + 118, y);
      y += 44;
    }
    return;
  }

  ctx.fillStyle = sc.mode === "dqn" ? C.accent : sc.mode === "optimal" ? C.good : C.warn;
  ctx.font = `700 42px ${FONT}`;
  ctx.fillText(sc.label, RX, y + 40);
  y += 92;

  ctx.fillStyle = C.dim;
  ctx.font = `24px ${FONT}`;
  const blurb = {
    optimal: ["規則只有兩條：", "① 最外面那個環隨時可以上下。",
              "② 環 k 要動，必須環 k−1 在桿上、", "　 而且比它更外面的全都不在。",
              "所以任何時候最多只有兩個環動得了。"],
    greedy: ["能拿下來就拿下來，", "不能拿才勉強裝回去。", "人第一次玩九連環都是這樣想的。"],
    random: ["每一步從合法動作裡亂抽一個。", "狀態圖是一條 512 格的直線，",
             "隨機遊走要走完得花 N² 步。"],
    dqn: ["它看到 18 個 0 跟 1，外加一個時間。", "沒有人告訴它格雷碼，",
          "也沒有人告訴它什麼時候該退。"],
  }[sc.mode] || [];
  blurb.forEach((l, i) => ctx.fillText(l, RX, y + i * 38));
  y += blurb.length * 38 + 30;

  const e = cur.engine;
  const cells = [["已走步數", e.steps.toLocaleString()], ["還要幾步", `${e.dist}`],
                 ["最優步數", `${e.startDistance}`],
                 ["多走了", e.solved ? `${e.steps - e.startDistance}` : "—"]];
  cells.forEach(([k, v], i) => {
    const cx = RX + (i % 2) * (RW / 2);
    const cy = y + Math.floor(i / 2) * 108;
    ctx.fillStyle = C.faint;
    ctx.font = `19px ${FONT}`;
    ctx.fillText(k, cx, cy);
    ctx.fillStyle = C.fg;
    ctx.font = `700 46px ${MONO}`;
    ctx.fillText(v, cx, cy + 50);
  });
  y += 216;

  if (e.over) {
    ctx.fillStyle = e.solved ? "rgba(74,222,128,0.12)" : "rgba(251,191,36,0.12)";
    roundRect(ctx, RX, y - 34, RW, 66, 8);
    ctx.fill();
    ctx.fillStyle = e.solved ? C.good : C.warn;
    ctx.font = `700 28px ${FONT}`;
    ctx.fillText(e.solved ? `解開了 · ${e.steps} 步` : `${e.steps} 步用完，還剩 ${e.dist} 步`, RX + 22, y + 8);
    y += 92;
  }

  if (sc.mode === "dqn" && cur.lastQ) {
    ctx.fillStyle = C.faint;
    ctx.font = `20px ${FONT}`;
    ctx.fillText("Q 值 — 牠覺得動哪個環比較划算", RX, y);
    y += 22;
    drawQBars(RX, y, RW, cur.lastQ, e.legal(), cur.lastA);
  }
}

function drawQBars(x, y, w, q, mask, chosen) {
  const lo = Math.min(...q), hi = Math.max(...q);
  const span = Math.max(1e-6, hi - lo);
  const x0 = x + 62, maxW = w - 62 - 108, barH = 22, gap = 8;
  for (let i = 0; i < q.length; i++) {
    const yy = y + i * (barH + gap);
    const legal = mask[i] === 1;
    ctx.fillStyle = legal ? (i === chosen ? C.fg : C.dim) : "#3a4657";
    ctx.font = `19px ${FONT}`;
    ctx.fillText(`環${i + 1}`, x, yy + barH - 4);
    // 不合法的動作在訓練時被壓成 -1e9，網路對它們輸出什麼都沒意義，
    // 所以這裡只畫一小截，不讓它們在畫面上跟真正的候選動作搶注意力。
    const bw = legal ? Math.max(4, ((q[i] - lo) / span) * maxW) : 26;
    ctx.fillStyle = i === chosen ? C.accent : legal ? "#334155" : "#161f30";
    roundRect(ctx, x0, yy, bw, barH, 4);
    ctx.fill();
    ctx.fillStyle = legal ? (i === chosen ? "#e0f2fe" : C.faint) : "#2c3648";
    ctx.font = `17px ${MONO}`;
    ctx.fillText(legal ? q[i].toFixed(3) : "不合法", x0 + bw + 10, yy + barH - 4);
  }
}

// ---------------------------------------------------------------- shaping 對照頁

function drawShaping(local) {
  const reveal = Math.min(1, local / (FPS * 1.6));
  ctx.textAlign = "left";
  ctx.fillStyle = C.fg;
  ctx.font = `700 50px ${FONT}`;
  ctx.fillText("照教科書寫，會學不起來", 110, 100);
  ctx.fillStyle = C.faint;
  ctx.font = `25px ${FONT}`;
  ctx.fillText("同一份程式、同一組超參數，只改 shaping 的一個常數。三條線都是真的跑出來的。", 110, 146);

  // 公式
  ctx.fillStyle = "#0d1626";
  roundRect(ctx, 110, 182, 1700, 108, 10);
  ctx.fill();
  ctx.fillStyle = C.accent;
  ctx.font = `700 32px ${MONO}`;
  ctx.fillText("F(s,s') = shaping × ( γs · Φ(s') − Φ(s) )        Φ(s) = −distance(s)", 146, 232);
  ctx.fillStyle = C.dim;
  ctx.font = `24px ${FONT}`;
  ctx.fillText("展開後多出一項 (1 − γs) × d。d 最大 341，γs = 0.95 時這一項有 17——把方向完全蓋掉。", 146, 272);

  // 三條曲線
  const box = { x: 110, y: 330, w: 1120, h: 500 };
  ctx.fillStyle = "#0b1020";
  roundRect(ctx, box.x, box.y, box.w, box.h, 12);
  ctx.fill();
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  ctx.stroke();

  const pad = { l: 96, r: 30, t: 54, b: 54 };
  const maxX = Math.max(...ablation.runs.flatMap((r) => r.history.map((h) => h.step)));
  const X = (v) => box.x + pad.l + (v / maxX) * (box.w - pad.l - pad.r);
  const Y = (v) => box.y + box.h - pad.b - v * (box.h - pad.t - pad.b);

  ctx.fillStyle = C.faint;
  ctx.font = `20px ${FONT}`;
  ctx.fillText("511 個起點裡，走出剛好最優的比例", box.x + 26, box.y + 34);
  ctx.font = `16px ${MONO}`;
  ctx.strokeStyle = "#16213a";
  for (let i = 0; i <= 4; i++) {
    const v = i / 4, yy = Y(v);
    ctx.beginPath();
    ctx.moveTo(box.x + pad.l, yy);
    ctx.lineTo(box.x + box.w - pad.r, yy);
    ctx.stroke();
    ctx.fillStyle = C.faint;
    ctx.textAlign = "right";
    ctx.fillText(`${(v * 100).toFixed(0)}%`, box.x + pad.l - 12, yy + 6);
  }
  ctx.textAlign = "left";
  ctx.fillText(`${(maxX / 1000).toFixed(0)}k transitions`, box.x + box.w - pad.r - 190, box.y + box.h - 18);

  const colors = [C.accent, C.bad, "#a78bfa"];
  ablation.runs.forEach((run, i) => {
    const pts = run.history.filter((h) => h.step <= maxX * reveal);
    if (pts.length < 2) return;
    ctx.strokeStyle = colors[i];
    ctx.lineWidth = 4;
    ctx.beginPath();
    pts.forEach((h, k) => (k ? ctx.lineTo(X(h.step), Y(h.optimal_rate)) : ctx.moveTo(X(h.step), Y(h.optimal_rate))));
    ctx.stroke();
  });

  // 圖例 + 結論
  let ly = box.y + 40;
  ablation.runs.forEach((run, i) => {
    ctx.fillStyle = colors[i];
    roundRect(ctx, 1290, ly - 20, 26, 26, 5);
    ctx.fill();
    ctx.fillStyle = C.fg;
    ctx.font = `700 26px ${FONT}`;
    ctx.fillText(run.name, 1332, ly);
    ctx.fillStyle = C.faint;
    ctx.font = `21px ${FONT}`;
    ctx.fillText(run.setting, 1332, ly + 34);
    ctx.fillStyle = run.final_optimal_rate > 0.5 ? C.good : C.bad;
    ctx.font = `700 34px ${MONO}`;
    ctx.fillText(`${(run.final_optimal_rate * 100).toFixed(1)}%`, 1332, ly + 78);
    ly += 152;
  });

  ctx.fillStyle = "#0d1626";
  roundRect(ctx, 110, 872, 1700, 130, 10);
  ctx.fill();
  ctx.fillStyle = C.accent;
  ctx.font = `700 30px ${FONT}`;
  ctx.fillText("理論保證的前提沒滿足的時候，要知道它為什麼沒滿足。", 146, 918);
  ctx.fillStyle = C.dim;
  ctx.font = `24px ${FONT}`;
  ctx.fillText("Ng 那篇論文要求 γs = γ 才保證最優策略不變。這裡把 γs 設成 1，換來的是「走對一步就加分」，", 146, 958);
  ctx.fillText("代價是理論保證消失——所以才要窮舉 511 個起點去驗，而不是相信它。", 146, 992);
}

// ---------------------------------------------------------------- 結果頁

function drawResults(local) {
  const reveal = Math.min(1, local / (FPS * 1.2));
  ctx.textAlign = "left";
  ctx.fillStyle = C.fg;
  ctx.font = `700 50px ${FONT}`;
  ctx.fillText(`從全部 ${bench.starts} 個起點各解一次`, 110, 104);
  ctx.fillStyle = C.faint;
  ctx.font = `24px ${FONT}`;
  ctx.fillText("環境是決定性的，狀態只有 512 個 —— 這是窮舉，不是抽樣。沒有信賴區間可談。", 110, 148);

  const cols = [
    { key: "solve_rate", title: "解得開的起點比例", max: 1, fmt: (v) => `${(v * 100).toFixed(1)}%` },
    { key: "optimal_rate", title: "走出剛好最優的比例", max: 1, fmt: (v) => `${(v * 100).toFixed(1)}%` },
    { key: "mean_ratio", title: "解開時平均走了最優的幾倍（越接近 1 越好）", max: 21, fmt: (v) => (v ? `${v.toFixed(2)}x` : "—") },
  ];
  const rows = bench.rows;
  const ROW_H = 46, GROUP_GAP = 18;
  let y = 200;
  for (const col of cols) {
    ctx.fillStyle = C.dim;
    ctx.font = `700 26px ${FONT}`;
    ctx.fillText(col.title, 110, y);
    y += 22;
    rows.forEach((r, i) => {
      const v = r[col.key];
      const yy = y + i * ROW_H;
      const isDqn = r.name.startsWith("DQN");
      ctx.fillStyle = isDqn ? C.fg : C.faint;
      ctx.font = `${isDqn ? 700 : 400} 23px ${FONT}`;
      ctx.fillText(r.name, 110, yy + 26);
      const x0 = 500, maxW = 1000;
      const frac = v == null ? 0 : Math.min(1, v / col.max);
      ctx.fillStyle = "#131c2e";
      roundRect(ctx, x0, yy + 7, maxW, 26, 6);
      ctx.fill();
      ctx.fillStyle = isDqn ? C.accent : r.name.startsWith("最優") ? C.good : "#475569";
      roundRect(ctx, x0, yy + 7, Math.max(4, maxW * frac * reveal), 26, 6);
      ctx.fill();
      ctx.fillStyle = isDqn ? "#e0f2fe" : C.dim;
      ctx.font = `700 25px ${MONO}`;
      ctx.fillText(col.fmt(v), x0 + maxW + 24, yy + 28);
    });
    y += rows.length * ROW_H + GROUP_GAP;
  }

  ctx.fillStyle = "#0d1626";
  roundRect(ctx, 110, 888, 1700, 116, 10);
  ctx.fill();
  ctx.fillStyle = C.accent;
  ctx.font = `700 30px ${FONT}`;
  ctx.fillText("DQN 走出來的路，跟格雷碼算出來的最優解一模一樣——511 個起點，一個都沒走錯。", 146, 934);
  ctx.fillStyle = C.dim;
  ctx.font = `24px ${FONT}`;
  ctx.fillText("但它沒有比較聰明：512 個狀態直接開一張表，五千局就收斂了。這裡值得學的是方法，不是結果。", 146, 974);
}

function drawProgress() {
  const p = frameIndex / plan.totalFrames;
  ctx.fillStyle = "#111a2c";
  ctx.fillRect(0, H - 5, W, 5);
  ctx.fillStyle = C.accent;
  ctx.fillRect(0, H - 5, W * p, 5);
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

window.__rec = { init, renderFrame, get plan() { return plan; } };
