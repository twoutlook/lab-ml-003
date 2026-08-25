// 九連環的純邏輯層：不碰 DOM、不畫圖。
// 這份規則必須跟 ml/env.py + ml/solver.py 一模一樣，AI 才有辦法把學到的東西搬回網頁。
//
// 環的編號：ring 1（索引 0）在最外側，可以自由上下；ring 9（索引 8）在最裡面。
// state[i] = 1 代表第 i+1 環還在桿上。目標：全 1 -> 全 0。
//
// 規則只有兩條：
//   1. ring 1 隨時可以動。
//   2. ring k 可以動，若且唯若 ring k-1 在桿上，而且 ring 1..k-2 全都不在。

export function legalMask(s) {
  const n = s.length;
  const m = new Int8Array(n);
  m[0] = 1;
  for (let k = 1; k < n; k++) {
    if (s[k - 1] !== 1) continue;
    let clearBelow = true;
    for (let j = 0; j < k - 1; j++) if (s[j]) { clearBelow = false; break; }
    if (clearBelow) { m[k] = 1; break; }   // 至多命中一個 k
  }
  return m;
}

// 還要幾步才解得完。把狀態當格雷碼解碼，就是剩餘步數。
export function distance(s) {
  const n = s.length;
  let b = 0, total = 0;
  for (let i = n - 1; i >= 0; i--) {
    b ^= s[i];
    if (b) total |= 1 << i;
  }
  return total;
}

// distance 的反函數。狀態圖是一條路徑，所以距離和狀態一對一。
export function stateFromDistance(d, n) {
  const s = new Int8Array(n);
  for (let i = 0; i < n; i++) {
    const bi = (d >> i) & 1;
    const bj = i + 1 < n ? (d >> (i + 1)) & 1 : 0;
    s[i] = bi ^ bj;
  }
  return s;
}

// 最優動作：走那個讓 distance 變小的合法動作。
export function optimalAction(s) {
  const d0 = distance(s);
  if (d0 === 0) return -1;
  const m = legalMask(s);
  for (let i = 0; i < s.length; i++) {
    if (!m[i]) continue;
    s[i] ^= 1;
    const d = distance(s);
    s[i] ^= 1;
    if (d === d0 - 1) return i;
  }
  return -1;
}

export class RingsEngine {
  constructor(cfg) {
    this.cfg = cfg;
    this.n = cfg.rings;
    this.maxSteps = cfg.maxSteps;
    this.fullDistance = distance(new Int8Array(this.n).fill(1));
    this.reset();
  }

  // start: "full"（全上）或一個 0..2^n-1 的距離值
  reset(start = "full") {
    this.s = start === "full"
      ? new Int8Array(this.n).fill(1)
      : stateFromDistance(start, this.n);
    this.startDistance = distance(this.s);
    this.dist = this.startDistance;
    this.steps = 0;
    this.illegal = 0;
    this.over = this.dist === 0;
    this.solved = this.over;
    return this.getObservation();
  }

  legal() { return legalMask(this.s); }

  // 19 維，排版跟 ml/env.py 的 obs() 逐位對齊
  getObservation() {
    const n = this.n;
    const o = new Float64Array(2 * n + 1);
    const m = this.legal();
    for (let i = 0; i < n; i++) {
      o[i] = this.s[i] * 2 - 1;      // 0/1 -> -1/+1
      o[n + i] = m[i];
    }
    o[2 * n] = this.steps / this.maxSteps;
    return o;
  }

  step(a) {
    if (this.over) return { ok: false, over: true };
    this.steps++;
    if (!this.legal()[a]) {
      this.illegal++;
      if (this.steps >= this.maxSteps) this.over = true;
      return { ok: false, over: this.over };
    }
    this.s[a] ^= 1;
    this.dist = distance(this.s);
    if (this.dist === 0) { this.solved = true; this.over = true; }
    else if (this.steps >= this.maxSteps) this.over = true;
    return { ok: true, over: this.over };
  }
}
