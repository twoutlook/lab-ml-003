/**
 * 把 web/record.html 錄成 MP4。
 *
 *   node tools/record_video.mjs [--fps 30] [--quality 0.92] [--url ...]
 *
 * 作法跟即時螢幕錄影不一樣：headless Chrome 一格一格地把畫面畫出來，
 * 每格用 canvas.toDataURL 取出來直接餵給 ffmpeg。所以不會掉格，
 * 也跟機器快慢無關 —— 同樣的 seed 一定錄出同樣的影片。
 *
 * 旁白不是「錄進去」的，是錄完之後照場景起始時間貼上去的，
 * 對時是算出來的，不是對出來的。
 *
 * 輸出：out/rings-dqn.mp4
 */
import puppeteer from "puppeteer-core";
import { spawn } from "node:child_process";
import { mkdirSync, existsSync, writeFileSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(
  path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1"), "..");
const OUT = path.join(ROOT, "out");
const VOICE = path.join(OUT, "voice");

const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i > -1 ? process.argv[i + 1] : d; };
const FPS = Number(arg("fps", 30));
const QUALITY = Number(arg("quality", 0.92));
const URL_ = arg("url", "http://localhost:8000/web/record.html");
const LIMIT = Number(arg("limit", 0));  // 只錄前 N 格，用來快速試版面
const SILENT = path.join(OUT, "silent.mp4");
const FINAL = path.join(OUT, arg("out", "rings-dqn.mp4"));

const CHROME = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, "Google/Chrome/Application/chrome.exe"),
].filter(Boolean).find(existsSync);

function run(cmd, args) {
  return new Promise((res, rej) => {
    const p = spawn(cmd, args, { stdio: ["ignore", "inherit", "inherit"] });
    p.on("error", rej);
    p.on("close", (c) => (c === 0 ? res() : rej(new Error(`${cmd} exited ${c}`))));
  });
}

async function main() {
  if (!CHROME) throw new Error("找不到 Chrome");
  mkdirSync(OUT, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--force-device-scale-factor=1",
           "--font-render-hinting=none", "--hide-scrollbars"],
  });
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.error("PAGE ERROR:", e.message));
  page.on("console", (m) => { if (m.type() === "error") console.error("console:", m.text()); });
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });

  console.log(`載入 ${URL_}`);
  await page.goto(URL_, { waitUntil: "networkidle0", timeout: 120000 });
  await page.waitForFunction("window.__rec !== undefined", { timeout: 60000 });

  console.log("初始化（會先把每一局模擬跑完，算出每格要推進幾步）…");
  const plan = await page.evaluate(() => window.__rec.init());
  writeFileSync(path.join(OUT, "plan.json"), JSON.stringify(plan, null, 2));
  console.log(`總長 ${plan.durationSec}s · ${plan.totalFrames} frames @ ${plan.fps}fps`);
  for (const s of plan.scenes) {
    const extra = s.kind === "game"
      ? `  seed=${s.seed} steps=${s.totalSteps} ≈${(s.stepsPerFrame * FPS / 60).toFixed(1)}×速` : "";
    console.log(`  ${s.id.padEnd(9)} ${(s.frames / FPS).toFixed(1).padStart(5)}s${extra}`);
  }

  const ff = spawn("ffmpeg", [
    "-y", "-hide_banner", "-loglevel", "error",
    "-f", "image2pipe", "-framerate", String(FPS), "-i", "-",
    "-c:v", "libx264", "-preset", "medium", "-crf", "19",
    "-pix_fmt", "yuv420p", "-movflags", "+faststart", SILENT,
  ], { stdio: ["pipe", "inherit", "inherit"] });

  const write = (buf) => new Promise((res) => (ff.stdin.write(buf) ? res() : ff.stdin.once("drain", res)));

  const total = LIMIT > 0 ? Math.min(LIMIT, plan.totalFrames) : plan.totalFrames;
  const t0 = Date.now();
  for (let i = 0; i < total; i++) {
    const dataUrl = await page.evaluate((q) => {
      window.__rec.renderFrame();
      return document.getElementById("stage").toDataURL("image/jpeg", q);
    }, QUALITY);
    await write(Buffer.from(dataUrl.slice(dataUrl.indexOf(",") + 1), "base64"));
    if (i % 150 === 0 || i === total - 1) {
      const el = (Date.now() - t0) / 1000;
      const eta = el / Math.max(1, i + 1) * (total - i - 1);
      process.stdout.write(`\r  frame ${i + 1}/${plan.totalFrames}  ${(el).toFixed(0)}s 已過，約剩 ${eta.toFixed(0)}s   `);
    }
  }
  process.stdout.write("\n");
  ff.stdin.end();
  await new Promise((res, rej) => ff.on("close", (c) => (c === 0 ? res() : rej(new Error(`ffmpeg ${c}`)))));
  await browser.close();
  console.log(`無聲影片 -> ${SILENT}`);

  if (LIMIT > 0) { console.log("--limit 模式，跳過配音"); return; }

  // 旁白照場景起點貼上去
  const inputs = [], filters = [], labels = [];
  plan.scenes.forEach((s, i) => {
    const mp3 = path.join(VOICE, `${s.id}.mp3`);
    if (!existsSync(mp3)) throw new Error(`缺旁白 ${mp3}，先跑 python tools/make_voice.py`);
    inputs.push("-i", mp3);
    const delayMs = Math.round((s.start / FPS) * 1000);
    filters.push(`[${i + 1}:a]adelay=${delayMs}|${delayMs}[a${i}]`);
    labels.push(`[a${i}]`);
  });
  const fc = `${filters.join(";")};${labels.join("")}amix=inputs=${labels.length}:normalize=0[aout]`;

  await run("ffmpeg", [
    "-y", "-hide_banner", "-loglevel", "error",
    "-i", SILENT, ...inputs,
    "-filter_complex", fc,
    "-map", "0:v", "-map", "[aout]",
    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    "-shortest", FINAL,
  ]);
  console.log(`完成 -> ${FINAL}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
