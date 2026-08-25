/** 抽幾格關鍵畫面存成 PNG，用來檢查版面，不用等整支影片錄完。
 *    node tools/preview_frames.mjs 20 900 1800 3200 4600
 */
import puppeteer from "puppeteer-core";
import { mkdirSync, existsSync, writeFileSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1"), "..");
const OUT = path.join(ROOT, "out", "preview");
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"].find(existsSync);

const targets = process.argv.slice(2).map(Number).filter((n) => Number.isFinite(n));
mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true,
  args: ["--no-sandbox", "--font-render-hinting=none", "--hide-scrollbars"] });
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("PAGE ERROR:", e.message));
await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
await page.goto("http://localhost:8000/web/record.html", { waitUntil: "networkidle0" });
await page.waitForFunction("window.__rec !== undefined");
const plan = await page.evaluate(() => window.__rec.init());

const max = Math.max(...targets);
for (let i = 0; i <= max; i++) {
  const url = await page.evaluate((want) => {
    window.__rec.renderFrame();
    return want ? document.getElementById("stage").toDataURL("image/png") : null;
  }, targets.includes(i));
  if (url) {
    const sc = plan.scenes.find((s) => i >= s.start && i < s.start + s.frames);
    const f = path.join(OUT, `f${String(i).padStart(5, "0")}_${sc ? sc.id : "end"}.png`);
    writeFileSync(f, Buffer.from(url.slice(url.indexOf(",") + 1), "base64"));
    console.log(f);
  }
}
await browser.close();
