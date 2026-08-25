"""把訓練好的 checkpoint 轉成網頁可以直接讀的 policy.json。

    python export_policy.py                 # 用 checkpoints/best.pt
    python export_policy.py --ckpt checkpoints/last.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import DuelingQNet

HERE = Path(__file__).resolve().parent


def dump_linear(lin: torch.nn.Linear, nd: int):
    w = [[round(float(v), nd) for v in row] for row in lin.weight.detach().cpu()]
    b = [round(float(v), nd) for v in lin.bias.detach().cpu()]
    return {"w": w, "b": b}  # w 是 [out][in]，跟 PyTorch 一樣


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "checkpoints" / "best.pt"))
    ap.add_argument("--out", default=str(HERE.parent / "web" / "policy.json"))
    ap.add_argument("--digits", type=int, default=5, help="權重保留幾位小數（越少檔案越小）")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = DuelingQNet(ck["obs_size"], ck["n_actions"], ck["hidden"])
    net.load_state_dict(ck["net"])
    net.eval()

    d = args.digits
    out = {
        "arch": "dueling_mlp",
        "obs_size": ck["obs_size"],
        "n_actions": ck["n_actions"],
        "trained_steps": ck.get("step", 0),
        "eval": ck.get("eval", None),
        "body": [dump_linear(net.body[0], d), dump_linear(net.body[2], d)],
        "value": [dump_linear(net.value[0], d), dump_linear(net.value[2], d)],
        "adv": [dump_linear(net.adv[0], d), dump_linear(net.adv[2], d)],
    }
    p = Path(args.out)
    p.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    mb = p.stat().st_size / 1e6
    print(f"寫出 {p}  ({mb:.2f} MB, trained_steps={out['trained_steps']:,})")
    if out["eval"]:
        e = out["eval"]
        print(f"  這份權重的 eval: 512 個起點解開 {e['solve_rate']:.1%}、"
              f"剛好最優 {e['optimal_rate']:.1%}，從全上出發用 {e['full_steps']} 步")
    print("  接著可以跑跨語言對帳：cd ../web && node _parity_test.mjs")


if __name__ == "__main__":
    main()
