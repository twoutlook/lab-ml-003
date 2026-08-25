"""用 Double DQN 訓練九連環 agent。

跑法（在 ml/ 目錄下）：
    python train.py                        # 預設 800,000 個 transition，幾分鐘
    python train.py --steps 200000         # 快速試跑
    python train.py --device cpu           # 沒 GPU 也能跑（這個環境很小，其實不慢）

想看「教材反例」的話：
    python train.py --shaping 0            # 關掉 reward shaping -> 幾乎學不起來
    python train.py --start full           # 每局都從全上開始 -> 只走得到 342/512 個狀態
    python train.py --naive 0.05           # 加上「拿環有獎、裝回去有罰」-> 被教壞

訓練中會持續寫出：
    checkpoints/best.pt        目前 eval 最好的權重
    checkpoints/last.pt        最新權重（可續訓）
    checkpoints/log.csv        訓練曲線原始資料
    ../web/training_log.json   給網頁畫圖用
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from env import VecRings, load_config, mask_from_obs
from evaluate import eval_all_starts
from model import NEG_INF, DuelingQNet, NStepAccumulator, ReplayBuffer

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"
WEB = HERE.parent / "web"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=800_000, help="總共要收集幾個 transition")
    p.add_argument("--envs", type=int, default=64, help="同時跑幾個環境（越多 GPU 越划算）")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=None, help="預設讀 shared/config.json，才能跟 env 的 shaping 一致")
    p.add_argument("--n-step", type=int, default=5)
    p.add_argument("--buffer", type=int, default=300_000)
    p.add_argument("--learn-start", type=int, default=20_000, help="收集夠這麼多 transition 才開始更新網路")
    p.add_argument("--train-per-iter", type=int, default=2, help="每輪互動做幾次梯度更新")
    p.add_argument("--target-sync", type=int, default=1_000, help="每幾次梯度更新同步一次 target network")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.03)
    p.add_argument("--eps-decay-frac", type=float, default=0.30, help="epsilon 在前幾成的訓練量內衰減完")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--start", choices=["random", "full", "curriculum"], default="random",
                   help="每局從哪裡開始。random 會把 512 個狀態都走過，full 只走得到 342 個")
    p.add_argument("--shaping", type=float, default=None, help="覆蓋 config 的 reward.shaping（0 = 純稀疏獎勵）")
    p.add_argument("--shaping-gamma", type=float, default=None,
                   help="覆蓋 config 的 shapingGamma。設成跟 gamma 一樣就是教科書寫法——會學不起來，見 README")
    p.add_argument("--tag", default=None,
                   help="把產出寫到 checkpoints/<tag>/，而且不覆蓋網頁用的 training_log.json。跑對照實驗用")
    p.add_argument("--naive", type=float, default=None, help="覆蓋 config 的 reward.naiveRingOff（會把 agent 教壞的獎勵）")
    p.add_argument("--eval-every", type=int, default=25_000, help="每幾個 transition 評估一次")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true", help="從 checkpoints/last.pt 續訓")
    return p.parse_args()


def main():
    args = parse_args()
    global CKPT
    if args.tag:
        CKPT = CKPT / args.tag
    CKPT.mkdir(parents=True, exist_ok=True)
    WEB.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = load_config()
    if args.shaping is not None:
        cfg["reward"]["shaping"] = args.shaping
    if args.naive is not None:
        cfg["reward"]["naiveRingOff"] = args.naive
    if args.shaping_gamma is not None:
        cfg["shapingGamma"] = args.shaping_gamma
    gamma = float(cfg["gamma"]) if args.gamma is None else args.gamma

    start_max = 8 if args.start == "curriculum" else None
    vec = VecRings(args.envs, cfg=cfg, seed=args.seed, start=args.start, start_max=start_max)
    obs_size, n_actions = vec.obs_size, vec.n_actions
    full_dist = vec.envs[0].full_distance
    print(f"device={device}  obs_size={obs_size}  n_actions={n_actions}  envs={args.envs}")
    print(f"start={args.start}  shaping={cfg['reward']['shaping']}  shapingGamma={cfg.get('shapingGamma', 1.0)}  "
          f"naive={cfg['reward']['naiveRingOff']}  gamma={gamma}  最優步數={full_dist}")

    net = DuelingQNet(obs_size, n_actions, args.hidden).to(device)
    target = DuelingQNet(obs_size, n_actions, args.hidden).to(device)
    target.load_state_dict(net.state_dict())
    target.eval()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"參數量 {sum(p.numel() for p in net.parameters()):,}")

    start_step = 0
    if args.resume and (CKPT / "last.pt").exists():
        ck = torch.load(CKPT / "last.pt", map_location=device, weights_only=False)
        net.load_state_dict(ck["net"])
        target.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        print(f"從 last.pt 續訓，step={start_step:,}")

    buf = ReplayBuffer(args.buffer, obs_size, device)
    acc = NStepAccumulator(args.envs, args.n_step, gamma)

    obs = vec.reset()
    eps_decay_steps = max(1, int(args.steps * args.eps_decay_frac))
    step = start_step
    grad_steps = 0
    best_score = -1.0
    ep_solved = []
    history = []
    t0 = time.time()
    next_eval = step + args.eval_every

    log_path = CKPT / "log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    if new_log:
        log_w.writerow(["step", "eps", "train_solve", "solve_rate", "optimal_rate",
                        "full_steps", "mean_ratio", "start_max", "loss", "sps"])

    while step < args.steps:
        eps = max(args.eps_end, args.eps_start + (args.eps_end - args.eps_start) * (step / eps_decay_steps))

        # --- 選動作：epsilon-greedy，但只在「合法動作」裡挑 ---
        # 不做 masking 的話，網路有 7/9 的輸出是永遠用不到的垃圾，
        # 而且探索會有一大半浪費在根本動不了的環上。
        mask = mask_from_obs(obs)
        with torch.no_grad():
            q = net(torch.as_tensor(obs, device=device))
            q = q.masked_fill(torch.as_tensor(mask, device=device) < 0.5, NEG_INF)
            greedy = q.argmax(dim=1).cpu().numpy()
        # 從合法動作裡均勻抽一個：亂數乘上 mask 再取 argmax，不用寫迴圈
        rand = (rng.random(mask.shape) * mask).argmax(axis=1)
        actions = np.where(rng.random(args.envs) < eps, rand, greedy)

        # --- 跟環境互動 ---
        next_obs, rew, done, infos = vec.step(actions)
        for i in range(args.envs):
            for (s, a, R, s2, d, k) in acc.push(i, obs[i], actions[i], rew[i], next_obs[i], bool(done[i])):
                buf.add(s, a, R, s2, d, k)
            if done[i]:
                ep_solved.append(1.0 if infos[i].get("solved") else 0.0)
        obs = next_obs
        step += args.envs

        # --- 學習 ---
        loss_val = float("nan")
        for _ in range(args.train_per_iter if len(buf) >= args.learn_start else 0):
            s, a, R, s2, d, k = buf.sample(args.batch, rng)
            with torch.no_grad():
                # Double DQN：用線上網路「選」動作，用 target 網路「評價」它。
                # 兩邊都要先套 s2 的合法遮罩——遮罩就藏在 obs 裡（第 n..2n-1 維），不必另外存。
                m2 = mask_from_obs(s2)
                a2 = net(s2).masked_fill(m2 < 0.5, NEG_INF).argmax(dim=1, keepdim=True)
                q2 = target(s2).gather(1, a2).squeeze(1)
                y = R + (gamma ** k) * (1.0 - d) * q2
            q_pred = net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_pred, y)  # Huber：對離群的 TD error 比較不敏感
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            grad_steps += 1
            loss_val = float(loss.item())
            if grad_steps % args.target_sync == 0:
                target.load_state_dict(net.state_dict())

        # --- 評估 / 紀錄 ---
        if step >= next_eval:
            next_eval += args.eval_every
            ev = eval_all_starts(net, cfg, device)
            train_solve = float(np.mean(ep_solved[-200:])) if ep_solved else 0.0
            sps = (step - start_step) / max(1e-9, time.time() - t0)
            fs = ev["full_steps"]
            fs_txt = str(fs) if fs > 0 else "--"
            print(f"step {step:>9,}  eps {eps:.3f}  train_solve {train_solve:5.1%}  "
                  f"| 512 起點: 解開 {ev['solve_rate']:5.1%} 最優 {ev['optimal_rate']:5.1%}  "
                  f"全上 {fs_txt:>4} 步  loss {loss_val:.4f}  {sps:,.0f} steps/s")
            log_w.writerow([step, round(eps, 4), round(train_solve, 4), round(ev["solve_rate"], 4),
                            round(ev["optimal_rate"], 4), fs, round(ev["mean_ratio"], 4),
                            vec.envs[0].start_max, round(loss_val, 5), int(sps)])
            log_f.flush()
            history.append({"step": step, "eps": round(eps, 4),
                            "train_solve": round(train_solve, 3),
                            "solve_rate": round(ev["solve_rate"], 3),
                            "optimal_rate": round(ev["optimal_rate"], 3),
                            "full_steps": fs})
            if args.tag:
                (CKPT / "history.json").write_text(json.dumps(history), encoding="utf-8")
            else:
                (WEB / "training_log.json").write_text(json.dumps(history), encoding="utf-8")

            # curriculum：解得夠好就把起點往後推
            if args.start == "curriculum" and train_solve > 0.9:
                new_max = min(full_dist, int(vec.envs[0].start_max * 1.6) + 1)
                if new_max != vec.envs[0].start_max:
                    vec.set_start_max(new_max)
                    print(f"  -> curriculum 起點距離放寬到 {new_max}")

            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step,
                        "obs_size": obs_size, "n_actions": n_actions, "hidden": args.hidden},
                       CKPT / "last.pt")
            # 主指標用 optimal_rate（512 個起點裡有幾成走出剛好最優的路）。
            # 用 >= 而不是 >：打平時偏好比較新的權重。
            score = ev["optimal_rate"]
            if score >= best_score:
                best_score = score
                torch.save({"net": net.state_dict(), "step": step, "eval": ev,
                            "obs_size": obs_size, "n_actions": n_actions, "hidden": args.hidden},
                           CKPT / "best.pt")
                print(f"  -> 新紀錄，存成 best.pt (optimal_rate={best_score:.1%}, 全上 {fs_txt} 步)")

    log_f.close()
    print(f"\n訓練結束。最佳 optimal_rate = {best_score:.1%}（512 個起點）")
    print("接著跑：python export_policy.py   把權重轉成網頁用的 policy.json")


if __name__ == "__main__":
    main()
