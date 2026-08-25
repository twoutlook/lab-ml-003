// 在瀏覽器裡跑 policy.json 的前向傳播。
// 網路很小（~138k 參數），純 JS 迴圈就夠快，不需要 TensorFlow.js。

function linear(x, layer) {
  const { w, b } = layer;            // w 是 [out][in]，跟 PyTorch 一樣
  const out = new Float64Array(w.length);
  for (let o = 0; o < w.length; o++) {
    const row = w[o];
    let s = b[o];
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}

function relu(x) {
  for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0;
  return x;
}

export class Policy {
  constructor(spec) {
    this.spec = spec;
    this.obsSize = spec.obs_size;
    this.nActions = spec.n_actions;
    this.trainedSteps = spec.trained_steps || 0;
    this.eval = spec.eval || null;
  }

  static async load(url = "policy.json") {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`policy.json 讀不到 (${res.status})`);
    return new Policy(await res.json());
  }

  // 回傳 9 個動作的 Q 值（還沒套遮罩）
  qValues(obs) {
    let h = relu(linear(obs, this.spec.body[0]));
    h = relu(linear(h, this.spec.body[1]));
    const v = linear(relu(linear(h, this.spec.value[0])), this.spec.value[1])[0];
    const a = linear(relu(linear(h, this.spec.adv[0])), this.spec.adv[1]);
    let mean = 0;
    for (let i = 0; i < a.length; i++) mean += a[i];
    mean /= a.length;
    // Dueling: Q = V + A - mean(A)
    const q = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) q[i] = v + a[i] - mean;
    return q;
  }

  // mask 是必要的：訓練時就有遮罩，推論時沒遮就會選到根本動不了的環。
  // 這裡的 -1e9 要跟 ml/model.py 的 NEG_INF 一致。
  act(obs, mask) {
    const q = this.qValues(obs);
    let best = -1, bestV = -Infinity;
    for (let i = 0; i < q.length; i++) {
      const v = mask && !mask[i] ? -1e9 : q[i];
      if (v > bestV) { bestV = v; best = i; }
    }
    return { action: best, q };
  }
}
