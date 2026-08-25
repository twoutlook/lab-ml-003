// 跨語言對帳：用 JS 引擎 + JS 前向傳播跑同一份權重。
//
// 這個專案的對帳比打磚塊嚴格得多——環境完全沒有隨機性，
// 所以 JS 端的數字必須跟 Python 端「一模一樣」，差一步都算 bug。
//
//   cd web && node _parity_test.mjs

import fs from "fs";
import { RingsEngine, distance, legalMask, optimalAction, stateFromDistance } from "./engine.js";
import { Policy } from "./nn.js";

const root = new URL("..", import.meta.url).pathname.replace(/^\//, "");
const cfg = JSON.parse(fs.readFileSync(root + "shared/config.json", "utf8"));
const n = cfg.rings;

// --- 1. 規則自我檢查：距離 <-> 狀態必須是一對一 ---
const seen = new Set();
for (let d = 0; d < (1 << n); d++) seen.add(distance(stateFromDistance(d, n)));
const bijection = seen.size === (1 << n);
const fullDist = distance(new Int8Array(n).fill(1));
console.log(`規則檢查: distance/stateFromDistance 一對一 = ${bijection}，全上距離 = ${fullDist}（應為 341）`);

// --- 2. 最優解真的走得完嗎 ---
{
  const e = new RingsEngine(cfg);
  while (!e.over) e.step(optimalAction(e.s));
  console.log(`最優解: ${e.steps} 步，solved=${e.solved}`);
}

// --- 3. 用 policy.json 從全部 512 個起點各跑一次 ---
let policy;
try {
  policy = new Policy(JSON.parse(fs.readFileSync(root + "web/policy.json", "utf8")));
} catch {
  console.log("\n（還沒有 policy.json，先跑 ml/train.py 再跑 ml/export_policy.py）");
  process.exit(0);
}

let solved = 0, optimal = 0, live = 0, fullSteps = -1;
const t0 = Date.now();
let totalSteps = 0;
for (let d = 1; d < (1 << n); d++) {
  const e = new RingsEngine(cfg);
  e.reset(d);
  live++;
  while (!e.over) {
    const { action } = policy.act(e.getObservation(), e.legal());
    e.step(action);
  }
  totalSteps += e.steps;
  if (e.solved) {
    solved++;
    if (e.steps === d) optimal++;
    if (d === fullDist) fullSteps = e.steps;
  }
}
const pct = (x) => (x * 100).toFixed(1) + "%";
console.log(`\npolicy.json trained_steps = ${policy.trainedSteps.toLocaleString()}`);
if (policy.eval) {
  console.log(`Python 端: 解開 ${pct(policy.eval.solve_rate)}  最優 ${pct(policy.eval.optimal_rate)}  全上 ${policy.eval.full_steps} 步`);
}
console.log(`JS     端: 解開 ${pct(solved / live)}  最優 ${pct(optimal / live)}  全上 ${fullSteps} 步`);
console.log(`推論速度: ${(totalSteps / ((Date.now() - t0) / 1000)).toFixed(0)} steps/s (單執行緒 JS)`);

if (policy.eval) {
  const same =
    Math.abs(policy.eval.solve_rate - solved / live) < 1e-9 &&
    Math.abs(policy.eval.optimal_rate - optimal / live) < 1e-9 &&
    policy.eval.full_steps === fullSteps;
  console.log(same
    ? "\n對帳通過：兩邊逐位相同。"
    : "\n對帳失敗：engine.js 和 env.py 的規則跑掉了，或 nn.js 沒套遮罩。");
  process.exit(same ? 0 : 1);
}
