"""
train_2048.py — Train a TRM model to play 2048.

Usage:
    python scripts/train_2048.py [--n_games 5000] [--epochs 10] [--batch_size 256]

Output:
    outputs/2048_trm.pt  →  {"model": ..., "emb": ..., "acc": float, "config": {...}}
"""
import argparse
import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.game2048 import generate_training_data, Game2048Dataset, VOCAB_SIZE

# ── TRM Architecture (inline copy from demo_app.py) ───────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.w

class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        h = int(8/3 * d)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attn(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.dk ** 0.5), dim=-1)
        return self.o((attn @ v).transpose(1, 2).reshape(B, L, D))

class Block(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.n1, self.a = RMSNorm(d), Attn(d, h)
        self.n2, self.f = RMSNorm(d), SwiGLU(d)
    def forward(self, x):
        x = x + self.a(self.n1(x))
        return x + self.f(self.n2(x))

class Net(nn.Module):
    def __init__(self, d=128, l=2, h=4):
        super().__init__()
        self.ls = nn.ModuleList([Block(d, h) for _ in range(l)])
        self.n = RMSNorm(d)
    def forward(self, x):
        for layer in self.ls: x = layer(x)
        return self.n(x)

class LatentRec(nn.Module):
    def __init__(self, net, d=128, n=4):
        super().__init__()
        self.net, self.n = net, n
        self.zp = nn.Linear(3 * d, d, bias=False)
        self.yp = nn.Linear(2 * d, d, bias=False)
    def forward(self, x, y, z):
        for _ in range(self.n):
            z = self.net(self.zp(torch.cat([x, y, z], dim=-1)))
        return self.net(self.yp(torch.cat([y, z], dim=-1))), z

class TRM(nn.Module):
    def __init__(self, d=128, l=2, h=4, n=4, T=3, v=4):
        super().__init__()
        self.T = T
        net = Net(d, l, h)
        self.rec = LatentRec(net, d, n)
        self.head = nn.Linear(d, v, bias=False)
    def forward(self, x, y, z):
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.rec(x, y, z)
        y, z = self.rec(x, y, z)
        return (y, z), self.head(y)

# ── Training ───────────────────────────────────────────────────────────────────
def train(args):
    print(f"Generating {args.n_games} games of training data (depth=2)...")
    data = generate_training_data(n_games=args.n_games, depth=2)
    print(f"Total samples: {len(data)}")

    random.shuffle(data)
    n_val = max(1, int(0.1 * len(data)))
    val_data = data[:n_val]
    train_data = data[n_val:]

    train_ds = Game2048Dataset(train_data)
    val_ds   = Game2048Dataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    cfg = {"d": 128, "l": 2, "h": 4, "n": 4, "T": 3, "v": 4}
    model = TRM(**cfg)
    emb   = nn.Embedding(VOCAB_SIZE, cfg["d"])

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(emb.parameters()),
        lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train(); emb.train()
        total_loss, n_correct, n_total = 0.0, 0, 0
        for batch in train_loader:
            tokens = batch["tokens"]   # [B, 16]
            labels = batch["label"]    # [B]

            x = emb(tokens)            # [B, 16, d]
            y = torch.zeros_like(x)
            z = torch.zeros_like(x)

            (_, _), y_hat = model(x, y, z)          # [B, 16, 4]
            logits = y_hat.mean(dim=1)               # [B, 4]  mean-pool positions
            loss   = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(emb.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total   += labels.size(0)

        train_acc  = n_correct / n_total
        train_loss = total_loss / n_total

        # ── Val ──
        model.eval(); emb.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"]
                labels = batch["label"]
                x = emb(tokens)
                y = torch.zeros_like(x)
                z = torch.zeros_like(x)
                (_, _), y_hat = model(x, y, z)
                logits = y_hat.mean(dim=1)
                preds  = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total   += labels.size(0)
        val_acc = v_correct / v_total

        scheduler.step()
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {train_loss:.4f} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc

    # ── Save ──
    os.makedirs("outputs", exist_ok=True)
    checkpoint = {
        "model":  model.state_dict(),
        "emb":    emb.state_dict(),
        "acc":    best_acc,
        "config": cfg,
    }
    torch.save(checkpoint, "outputs/2048_trm.pt")
    print(f"\nSaved outputs/2048_trm.pt  (best val acc: {best_acc:.3f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games",    type=int, default=5000)
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    train(args)
