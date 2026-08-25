// 純繪圖：把 engine 的狀態畫到 canvas 上。engine 完全不知道 canvas 存在。
//
// 畫法是簡化過的九連環：橫桿（劍）穿過還在桿上的環，
// 已經解下來的環會被抬高到桿的上方。每個環底下那根直桿代表它固定在底板上，
// 所以環只能上下，不能左右移動——這就是規則 2 的來源。

const ACCENT = "#38bdf8";
const OFF_Y = 52;

export function drawPuzzle(ctx, e, view, { lastAction = -1, flash = 0 } = {}) {
  const W = view.width, H = view.height;
  const barY = view.barY, gap = view.ringGap, R = view.ringRadius;
  const x = (i) => W - view.marginRight - i * gap;
  const baseY = barY + 104;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b1020";
  ctx.fillRect(0, 0, W, H);

  const mask = e.legal();

  // 底板
  ctx.fillStyle = "#1e293b";
  roundRect(ctx, 40, baseY, W - 80, 12, 4);
  ctx.fill();

  // 橫桿（劍）：左端有個握把
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 7;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(52, barY);
  ctx.lineTo(W - 34, barY);
  ctx.stroke();
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.ellipse(40, barY, 16, 22, 0, 0, Math.PI * 2);
  ctx.stroke();

  for (let i = 0; i < e.n; i++) {
    const cx = x(i);
    const on = e.s[i] === 1;
    const cy = on ? barY : barY - OFF_Y;
    const legal = mask[i] === 1;

    // 直桿：把環綁在底板上
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, baseY);
    ctx.lineTo(cx, cy + R);
    ctx.stroke();

    // 環
    const isFlash = i === lastAction && flash > 0;
    ctx.strokeStyle = isFlash ? "#fde047" : legal ? ACCENT : "#475569";
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.stroke();

    // 合法的環給一個上下箭頭提示
    if (legal) {
      ctx.fillStyle = "rgba(56,189,248,0.85)";
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(on ? "▲" : "▼", cx, cy - R - 8);
    }

    // 編號
    ctx.fillStyle = legal ? "#cbd5e1" : "#475569";
    ctx.font = "12px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText(String(i + 1), cx, baseY + 28);
  }
  ctx.textAlign = "left";

  // 上方狀態列
  ctx.fillStyle = "#64748b";
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText(`state ${Array.from(e.s).join("")}`, 20, 26);
  ctx.textAlign = "right";
  ctx.fillStyle = e.dist === 0 ? "#4ade80" : "#94a3b8";
  ctx.fillText(`還要 ${e.dist} 步 · 已走 ${e.steps} 步`, W - 20, 26);
  ctx.textAlign = "left";

  // 進度條：從起始距離縮到 0
  const p = 1 - e.dist / Math.max(1, e.startDistance);
  ctx.fillStyle = "#1e293b";
  roundRect(ctx, 20, 36, W - 40, 6, 3);
  ctx.fill();
  ctx.fillStyle = e.dist === 0 ? "#4ade80" : ACCENT;
  roundRect(ctx, 20, 36, Math.max(2, (W - 40) * Math.max(0, p)), 6, 3);
  ctx.fill();
}

export function drawOverlay(ctx, view, lines) {
  ctx.fillStyle = "rgba(11,16,32,0.86)";
  ctx.fillRect(0, 0, view.width, view.height);
  ctx.textAlign = "center";
  let y = view.height / 2 - (lines.length - 1) * 18;
  lines.forEach((l, i) => {
    ctx.fillStyle = i === 0 ? "#f8fafc" : "#94a3b8";
    ctx.font = i === 0 ? "700 26px system-ui, sans-serif" : "14px system-ui, sans-serif";
    ctx.fillText(l, view.width / 2, y);
    y += i === 0 ? 40 : 24;
  });
  ctx.textAlign = "left";
}

// AI 對 9 個動作的 Q 值長條圖。不合法的動作會標成灰色 ——
// 訓練時它們被壓成 -1e9，所以網路輸出的那個值其實沒有意義。
export function drawQBars(canvas, q, mask, chosen) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!q) {
    ctx.fillStyle = "#475569";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("（不是 DQN 模式，沒有 Q 值）", 10, H / 2);
    return;
  }
  const lo = Math.min(...q), hi = Math.max(...q);
  const span = Math.max(1e-6, hi - lo);
  const barH = 14, gap = 5, x0 = 34, maxW = W - x0 - 62;
  for (let i = 0; i < q.length; i++) {
    const y = 6 + i * (barH + gap);
    const legal = !mask || mask[i] === 1;
    ctx.fillStyle = legal ? "#cbd5e1" : "#475569";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(`環${i + 1}`, 4, y + 11);
    const w = Math.max(2, ((q[i] - lo) / span) * maxW);
    ctx.fillStyle = i === chosen ? ACCENT : legal ? "#334155" : "#1b2430";
    roundRect(ctx, x0, y, w, barH, 3);
    ctx.fill();
    ctx.fillStyle = legal ? (i === chosen ? "#e0f2fe" : "#64748b") : "#3a4657";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(legal ? q[i].toFixed(3) : "不合法", x0 + w + 6, y + 11);
  }
}

// 訓練曲線：兩條比例（0..100%）
export function drawCurve(canvas, history) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#0b1020";
  ctx.fillRect(0, 0, W, H);
  if (!history || history.length < 2) {
    ctx.fillStyle = "#475569";
    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText("還沒有訓練紀錄（跑 ml/train.py 就會出現）", 12, H / 2);
    return;
  }
  const pad = { l: 34, r: 10, t: 10, b: 22 };
  const maxX = Math.max(...history.map((h) => h.step));
  const X = (v) => pad.l + (v / maxX) * (W - pad.l - pad.r);
  const Y = (v) => H - pad.b - v * (H - pad.t - pad.b);

  ctx.strokeStyle = "#1e293b";
  ctx.fillStyle = "#64748b";
  ctx.font = "10px ui-monospace, monospace";
  for (let i = 0; i <= 4; i++) {
    const v = i / 4, y = Y(v);
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(W - pad.r, y);
    ctx.stroke();
    ctx.fillText((v * 100).toFixed(0) + "%", 6, y + 3);
  }
  ctx.fillText(`${(maxX / 1000).toFixed(0)}k steps`, W - pad.r - 58, H - 6);

  const line = (key, color, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    history.forEach((h, i) => (i ? ctx.lineTo(X(h.step), Y(h[key])) : ctx.moveTo(X(h.step), Y(h[key]))));
    ctx.stroke();
  };
  line("solve_rate", "#475569", 1.5);
  line("optimal_rate", ACCENT, 2);

  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = ACCENT;
  ctx.fillText("走出剛好最優的起點比例", pad.l + 6, pad.t + 12);
  ctx.fillStyle = "#64748b";
  ctx.fillText("解得開就算的比例", pad.l + 6, pad.t + 26);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
