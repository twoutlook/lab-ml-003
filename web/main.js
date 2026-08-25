import { RingsEngine, optimalAction } from "./engine.js";
import { Policy } from "./nn.js";
import { drawPuzzle, drawOverlay, drawQBars, drawCurve } from "./render.js";

const $ = (id) => document.getElementById(id);
const canvas = $("game");
const ctx = canvas.getContext("2d");

let cfg, view, engine, policy = null;
let mode = "human";        // human | dqn | optimal | random
let speed = 4;             // 一個畫面更新走幾步
let running = false;
let lastQ = null, lastMask = null, lastAction = -1, flash = 0;
let stats = { runs: 0, solved: 0, bestSteps: null };

async function boot() {
  cfg = await (await fetch("../shared/config.json", { cache: "no-store" })).json();
  view = cfg.view;
  canvas.width = view.width;
  canvas.height = view.height;
  engine = new RingsEngine(cfg);

  $("hudOptimal").textContent = engine.fullDistance;

  try {
    policy = await Policy.load();
    const e = policy.eval;
    $("policyInfo").textContent =
      `已載入 policy.json：訓練 ${policy.trainedSteps.toLocaleString()} steps` +
      (e ? `，512 個起點裡 ${(e.optimal_rate * 100).toFixed(1)}% 走出剛好最優，從全上出發 ${e.full_steps} 步` : "");
    $("policyInfo").className = "note ok";
  } catch {
    $("policyInfo").textContent = "還沒有 policy.json —— 先跑 ml/train.py，再跑 ml/export_policy.py。DQN 模式目前不可用。";
    $("policyInfo").className = "note warn";
    $("modeDqn").disabled = true;
  }

  loadCurve();
  bindUI();
  restart();
  requestAnimationFrame(loop);
}

async function loadCurve() {
  try {
    const h = await (await fetch("training_log.json", { cache: "no-store" })).json();
    drawCurve($("curve"), h);
  } catch {
    drawCurve($("curve"), null);
  }
}

function bindUI() {
  for (const [id, m] of [["modeHuman", "human"], ["modeDqn", "dqn"],
                         ["modeOptimal", "optimal"], ["modeRandom", "random"]]) {
    $(id).addEventListener("click", () => {
      mode = m;
      document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("active", b.id === id));
      restart();
    });
  }
  $("btnStart").addEventListener("click", toggle);
  $("btnReset").addEventListener("click", restart);
  $("btnStep").addEventListener("click", () => { running = false; oneStep(); draw(); });
  $("speed").addEventListener("input", (ev) => {
    speed = Number(ev.target.value);
    $("speedLabel").textContent = `${speed} 步/幀`;
  });
  $("startSel").addEventListener("change", restart);

  addEventListener("keydown", (ev) => {
    if (ev.key === " ") { ev.preventDefault(); toggle(); }
    else if (ev.key === "r" || ev.key === "R") restart();
    else if (ev.key >= "1" && ev.key <= "9") tryHuman(Number(ev.key) - 1);
  });

  // 點畫面上的環：換算成環的索引
  canvas.addEventListener("click", (ev) => {
    if (mode !== "human" || engine.over) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * view.width;
    const i = Math.round((view.width - view.marginRight - x) / view.ringGap);
    if (i >= 0 && i < engine.n) tryHuman(i);
  });
}

function tryHuman(i) {
  if (mode !== "human" || engine.over) return;
  const res = engine.step(i);
  lastAction = i;
  flash = res.ok ? 6 : 12;
  if (engine.over) finish();
  draw();
}

function toggle() {
  if (mode === "human") return;      // 人類模式沒有「播放」
  if (engine.over) { restart(); }
  running = !running;
  $("btnStart").textContent = running ? "暫停 (Space)" : "播放 (Space)";
}

function restart() {
  const sel = $("startSel").value;
  const start = sel === "full"
    ? "full"
    : Math.max(1, Math.floor(Math.random() * (1 << cfg.rings)));
  engine.reset(start);
  running = false;
  lastQ = null; lastMask = null; lastAction = -1; flash = 0;
  $("btnStart").disabled = mode === "human";
  $("btnStep").disabled = mode === "human";
  $("btnStart").textContent = "播放 (Space)";
  draw();
}

function finish() {
  running = false;
  stats.runs++;
  if (engine.solved) {
    stats.solved++;
    if (stats.bestSteps === null || engine.steps < stats.bestSteps) stats.bestSteps = engine.steps;
  }
  $("btnStart").textContent = "再來一次 (Space)";
}

function chooseAction() {
  if (mode === "optimal") return optimalAction(engine.s);
  if (mode === "random") {
    const m = engine.legal();
    const idx = [];
    for (let i = 0; i < m.length; i++) if (m[i]) idx.push(i);
    return idx[(Math.random() * idx.length) | 0];
  }
  const mask = engine.legal();
  const { action, q } = policy.act(engine.getObservation(), mask);
  lastQ = q;
  lastMask = mask;
  return action;
}

function oneStep() {
  if (engine.over || mode === "human") return;
  const a = chooseAction();
  lastAction = a;
  engine.step(a);
  if (engine.over) finish();
}

function loop() {
  if (running && !engine.over) {
    for (let i = 0; i < speed; i++) {
      oneStep();
      if (engine.over) break;
    }
  }
  if (flash > 0) flash--;
  draw();
  requestAnimationFrame(loop);
}

function draw() {
  drawPuzzle(ctx, engine, view, { lastAction, flash });
  if (engine.over) {
    drawOverlay(ctx, view, engine.solved
      ? ["解開了", `${engine.steps} 步（最優 ${engine.startDistance} 步）`,
         engine.steps === engine.startDistance ? "一步都沒多走" : `多走了 ${engine.steps - engine.startDistance} 步`]
      : ["超過步數上限", `${engine.steps} 步之後還剩 ${engine.dist} 步`, "按 R 重來"]);
  }

  $("hudSteps").textContent = engine.steps;
  $("hudLeft").textContent = engine.dist;
  $("hudStart").textContent = engine.startDistance;
  $("hudExtra").textContent = engine.solved ? engine.steps - engine.startDistance : "-";
  $("hudRuns").textContent = stats.runs;
  $("hudSolved").textContent = stats.runs ? `${stats.solved}/${stats.runs}` : "-";
  $("hudBest").textContent = stats.bestSteps === null ? "-" : stats.bestSteps;
  drawQBars($("qbars"), mode === "dqn" ? lastQ : null, lastMask, lastAction);
}

boot();
