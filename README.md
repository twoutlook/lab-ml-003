# lab-ml-003 — 九連環 × DQN

**影片**（4:17，繁中旁白）https://youtu.be/CQ-X3vbpqIk ·
**可互動圖文版**（中英雙語，網頁上的 agent 是真的在跑推論）
https://claude.ai/code/artifact/ab8383b4-9e64-47ae-8578-800957e11865

系列的第三個。001 是打磚塊、002 是俄羅斯方塊，這個是九連環。
一樣是「自己寫遊戲、自己寫環境、自己訓練 agent」，沒有 Gym / Stable-Baselines3。

前兩個練的是「反應」，這個練的是**稀疏獎勵**和**探索**——
九連環要走 341 步才拿得到唯一的那一分，而且中途一定要把已經拿下來的環再裝回去。

```
shared/config.json     規則與獎勵的唯一來源（JS 和 Python 都讀它）
web/                   謎題本體（Canvas，無框架）+ 瀏覽器端推論
  engine.js              規則邏輯（不碰 DOM）
  render.js              畫面
  nn.js                  policy.json 的前向傳播（純 JS，不用 TF.js）
  main.js                模式切換、輸入、HUD
  index.html
  _parity_test.mjs       跨語言對帳（這個專案要求逐位相同）
ml/                    訓練
  solver.py              規則核心 + 格雷碼最優解（沒有任何 ML）
  env.py                 跟 engine.js 同規則的 numpy 環境 + VecRings
  model.py               Dueling Q 網路、n-step replay buffer、動作遮罩
  train.py               Double DQN 訓練迴圈
  tabular.py             表格式 Q-learning（512 個狀態，可以整張印出來）
  evaluate.py            窮舉 512 個起點，跟亂走 / 貪心 / 最優解比
  benchmark.py           同樣的窮舉，但寫成 json 給影片和 artifact 讀
  ablation.py            reward shaping 的三組對照實驗
  export_policy.py       權重 -> web/policy.json
  _smoke.py              改完規則一定要跑的自我檢查
tools/                 影片與 artifact
  script.json            旁白稿（每一段的長度決定該場景要幾格 frame）
  make_voice.py          edge-tts -> out/voice/*.mp3 + timing.json
  record_video.mjs       headless Chrome 逐格離線算圖 -> out/rings-dqn.mp4
  build_artifact.mjs     把資料與權重灌進樣板 -> out/artifact.html
  publish_youtube.py     建 playlist、組描述、上傳
```

## 快速開始

```bash
# 0. 先確認規則沒壞
cd ml && python _smoke.py

# 1. 開網頁（一定要用 http，file:// 不能 fetch）
python -m http.server 8000
#    瀏覽器打開 http://localhost:8000/web/

# 2. 訓練（RTX 4070 Ti SUPER 上約 3 分鐘）
cd ml
python train.py

# 3. 匯出給網頁，重整頁面就能看 AI 解
python export_policy.py

# 4. 跟基準線比
python evaluate.py

# 5. 跨語言對帳（這個專案要求 JS 和 Python 逐位相同）
cd ../web && node _parity_test.mjs
```

沒有 GPU：`python train.py --device cpu`。這個環境很小，CPU 也跑得動。

## 規則（只有兩條）

環從外到內編號 1..9，狀態用 9 個 bit 表示（1 = 還在桿上）。目標：全 1 變全 0。

1. 環 1 隨時可以上或下。
2. 環 k（k ≥ 2）可以動，若且唯若環 k−1 在桿上，而且環 1…k−2 全都不在。

推論出來的三件事，決定了這個專案的一切：

* **任何狀態最多只有 2 個合法動作。** 所以整張狀態圖是一條路徑：
  2⁹ = 512 個狀態排成一直線，你只能往前或往後一步。
* **全上的狀態離終點 341 步**（(2¹⁰−1)/3）。距離可以用格雷碼直接算出來，
  見 `solver.py` 的 `distance()`。也就是說——**這個問題有精確的最優解可以當基準**。
* **貪心必死。** 341 步裡有一半是在把環裝回去。`evaluate.py` 裡的「貪心拿環」基準線
  只解得開 0.2% 的起點。

## 這個專案在教什麼

| 概念 | 在哪裡 |
|---|---|
| 稀疏獎勵為什麼難 | 把 `reward.shaping` 設 0，341 步只有最後一步有分 |
| potential-based reward shaping | `env.py` 的 `Φ(s) = -distance(s)` |
| **shaping 的理論陷阱** | `env.py` 裡 `shapingGamma` 那段註解——照教科書寫會學不起來 |
| 動作遮罩（action masking） | `model.py` 的 `masked_q()`、`train.py` 的 `NEG_INF` |
| gamma 該設多少 | shaping 夠好時 gamma 可以壓到 0.95；shaping 關掉就得拉到 0.999 |
| 起點分佈就是課程設計 | `train.py --start random / full / curriculum` |
| 表格 vs 函數逼近 | `tabular.py` 幾秒收斂，DQN 要幾分鐘——但表格開不了大問題 |
| 窮舉式評估 | 512 個狀態全部跑一遍，沒有取樣雜訊 |
| 模型部署（跨語言搬權重） | `export_policy.py` -> `web/nn.js`，對帳要求逐位相同 |

## 狀態表示（19 維）

| 索引 | 內容 |
|---|---|
| 0–8 | 9 個環在不在桿上（0/1 映射成 −1/+1） |
| 9–17 | 這 9 個環現在動不動得了（合法遮罩） |
| 18 | 已經走了幾步 / 步數上限 |

動作有 9 個：`i` = 動第 i+1 個環。不合法的在選動作時就被壓成 −1e9。

索引 9–17 其實是冗餘的（從 0–8 就算得出來），放進去有兩個好處：
網路少學一件事，而且 replay buffer 可以直接從 `obs` 切出遮罩，不用另外存一份
（見 `env.py` 的 `mask_from_obs`）。

## 那個真的會踩到的坑

`env.py` 用的是 potential-based shaping：

```
F(s, s') = shaping * (γ_s * Φ(s') − Φ(s))，   Φ(s) = −distance(s)
```

教科書（Ng et al. 1999）說 γ_s 要等於訓練用的 gamma，這樣**最優策略保證不變**。
照做的結果是完全學不起來，原因值得記住：

Φ 和距離成正比，展開之後會多出一項 `(1 − γ_s) · d`。d 最大 341，
所以 γ_s = 0.95 時這一項就有 17，是真正帶方向的那個 ±1 的十七倍。
最後變成**走錯方向也拿正分**，agent 當然學不到方向。

這裡的做法是 `shapingGamma = 1.0`，讓 F 剛好等於 ±shaping。
代價是理論上不再保證最優策略不變——但 step cost 已經把「越短越好」寫進去了，
而且 `evaluate.py` 會逐一驗證 512 個起點，實際跑出來就是 100% 最優。

**理論保證的前提沒滿足的時候，要知道它為什麼沒滿足。**

## 實際跑出來的數字

`python evaluate.py`（512 個起點全部跑一次，環境是決定性的，沒有雜訊）：

| policy | 全上解得開 | 用幾步 | 解開比例 | 剛好最優 | 平均倍率 |
|---|---|---|---|---|---|
| 亂走（合法動作均勻） | 否 | — | 4.7% | 0.6% | 20.55x |
| 貪心拿環（只想往前） | 否 | — | 0.2% | 0.2% | 1.00x |
| 最優解（格雷碼） | 是 | 341 | 100.0% | 100.0% | 1.00x |
| DQN | 是 | 341 | 100.0% | 100.0% | 1.00x |

「亂走」在 1200 步內只解得開 4.7%，而且那幾乎全是本來就快到終點的起點——
因為在一條 512 長的路徑上隨機遊走，走到端點要 O(N²) 步。
這就是為什麼稀疏獎勵在這裡是真的難，不是假難。

## shaping 對照實驗（README 上那三句話是真的跑出來的）

```bash
cd ml && python ablation.py        # 三組各跑 30 萬 transition，約 4 分鐘
```

| 設定 | 差別 | 最後走出最優的起點比例 |
|---|---|---|
| `shapingGamma = 1` | 本專案的做法 | **100.0%** |
| `shapingGamma = γ` | 教科書寫法（Ng et al. 1999） | 0.0% |
| `shaping = 0` | 純稀疏獎勵，γ = 0.999 | 0.0% |

三組的程式與超參數完全一樣，只差 shaping 的一個常數。
結果寫進 `ml/checkpoints/ablation.json`，影片和 artifact 的那三條曲線就是讀這個檔案。

## 三個可以直接跑的對照實驗

```bash
# 1. 關掉 shaping，退回純稀疏獎勵（記得把 gamma 拉高，不然 341 步外的 +1 完全看不到）
python train.py --shaping 0 --gamma 0.999

# 2. 每局都從全上開始。只走得到 512 個狀態裡的 342 個，
#    另外 170 個永遠沒看過——訓練分佈決定了泛化範圍。
python train.py --start full

# 3. 加上「拿環有獎、裝回去有罰」的直覺獎勵。這是一個會主動把 agent 教壞的 reward。
python train.py --naive 0.05
```

每個都會在 `checkpoints/log.csv` 留下曲線，可以直接對照。

## 兩邊規則必須一致

`web/engine.js` 和 `ml/env.py` + `ml/solver.py` 是同一套規則的兩份實作。
改了其中一邊，另一邊一定要跟著改。

跟打磚塊不同的是，這個環境**完全沒有隨機性**，所以對帳可以要求逐位相同：

```bash
cd web && node _parity_test.mjs
```

它會用 JS 引擎跑 `policy.json` 走過全部 512 個起點，
解開比例、最優比例、從全上出發的步數，三個數字都必須跟 Python 端一模一樣。
差一步就是 bug。

## 接下來可以玩的

1. **把環數改成 11**（`shared/config.json` 的 `rings`）。最優步數變成 1365，
   狀態變 2048 個。同一組超參數還學得起來嗎？`maxSteps` 要跟著調。
2. **拿掉 obs 裡的合法遮罩（索引 9–17）**，只留 9 個 bit。網路自己學得會規則嗎？
3. **訓練 n=9，直接拿去解 n=7**。網路的輸入維度固定，所以要先想清楚怎麼對齊——
   這就是為什麼很多 RL 模型沒辦法換問題規模。
4. **用 curriculum 取代 shaping**：`--shaping 0 --start curriculum --gamma 0.999`。
   從離終點 8 步開始，解得動就往後推。比 shaping 慢，但不需要知道 distance 函數。
5. **換演算法**：這個環境的動作是離散的、狀態是有限的，很適合拿來比較
   DQN / PPO / MCTS 的樣本效率。

## 影片是怎麼做出來的

```bash
cd ml && python benchmark.py && python ablation.py    # 影片要用的數字
cd .. && python tools/make_voice.py                   # 旁白 + 每段長度
python -m http.server 8000                            # record.html 要用 http
node tools/record_video.mjs                           # -> out/rings-dqn.mp4
node tools/build_artifact.mjs                         # -> out/artifact.html
python tools/publish_youtube.py --dry-run             # 先看描述
python tools/publish_youtube.py                       # 上傳（預設 unlisted）
```

畫面不是螢幕錄影，是 headless Chrome 一格一格算出來再餵給 ffmpeg，所以不會掉格，
也跟機器快慢無關。旁白不是錄進去的，是錄完之後照場景起始時間貼上去的——
每個場景要幾格 frame，由那一段旁白的實際長度決定，對時是算出來的不是對出來的。

影片裡的每一個數字都從 `benchmark.json` / `ablation.json` / `plan.json` 讀，
沒有一個是手打進去的。改了訓練結果重錄一次，數字自己會跟著變。
