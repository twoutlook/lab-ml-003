/** 把訓練曲線、benchmark、shaping 對照實驗、以及整份權重灌進 artifact 樣板，產出單一 HTML。
 *    node tools/build_artifact.mjs
 *  輸出 out/artifact.html
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1"), "..");
const rd = (p) => readFileSync(path.join(ROOT, p), "utf8");
const rj = (p) => JSON.parse(rd(p));

const policy = rj("web/policy.json");
const bench = rj("ml/checkpoints/benchmark.json");
const ablation = rj("ml/checkpoints/ablation.json");
const history = rj("web/training_log.json");
const config = rj("shared/config.json");

const DATA = {
  config,
  history: history.map((h) => ({ step: h.step, solve_rate: h.solve_rate, optimal_rate: h.optimal_rate })),
  benchmark: bench,
  ablation,
  trained_steps: policy.trained_steps,
};

// 影片連結：有 out/youtube.json 就填進去，沒有就把整張卡拿掉，
// 不要留一個指向 __YT_URL__ 的死連結。
let yt = null;
try { yt = rj("out/youtube.json"); } catch {}
let tpl = rd("tools/artifact_template.html");
if (yt && yt.video_url) {
  tpl = tpl.replaceAll("__YT_URL__", yt.video_url);
} else {
  tpl = tpl.replace(/<!--YT_START-->[\s\S]*?<!--YT_END-->/, "");
}

const out = tpl
  .replace("/*__DATA__*/", JSON.stringify(DATA))
  .replace("/*__POLICY__*/", JSON.stringify({
    obs_size: policy.obs_size, n_actions: policy.n_actions,
    body: policy.body, value: policy.value, adv: policy.adv,
  }));

if (out.includes("/*__DATA__*/") || out.includes("/*__POLICY__*/")) {
  throw new Error("樣板裡的占位符沒被換掉——檢查 artifact_template.html");
}

mkdirSync(path.join(ROOT, "out"), { recursive: true });
const dst = path.join(ROOT, "out", "artifact.html");
writeFileSync(dst, out, "utf8");
console.log(`${dst}  ${(Buffer.byteLength(out) / 1e6).toFixed(2)} MB`);
console.log(`  影片連結  ${yt && yt.video_url ? yt.video_url : "（無，已移除該張卡）"}`);
console.log(`  benchmark ${bench.starts} 個起點 · 訓練曲線 ${history.length} 點 · ` +
            `對照實驗 ${ablation.runs.length} 組 · 權重 ${policy.trained_steps.toLocaleString()} steps`);
