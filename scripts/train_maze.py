"""
Train TRM maze model on locally-generated 10x10 mazes.

Key improvements over the original Colab notebook:
  - Full path lengths (no 30-step cap)
  - Configurable T (default 4, vs original 2)
  - Saves checkpoint compatible with demo_app.py

Usage:
    cd C:/Users/bhure/.gemini/antigravity/scratch/TRM
    python scripts/train_maze.py                        # T=4, 20k iters
    python scripts/train_maze.py --T 6 --iters 30000
    python scripts/train_maze.py --T 2 --iters 10000 --device cpu
"""

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset


# ── Compact TRM architecture (matches demo_app.py) ───────────────────────────

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
        h = int(8 / 3 * d)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attn(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.dh ** 0.5), dim=-1)
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
    def __init__(self, d=256, l=2, h=4):
        super().__init__()
        self.ls = nn.ModuleList([Block(d, h) for _ in range(l)])
        self.n = RMSNorm(d)
    def forward(self, x):
        for layer in self.ls:
            x = layer(x)
        return self.n(x)


class LatentRec(nn.Module):
    def __init__(self, net, d=256, n=4):
        super().__init__()
        self.net, self.n = net, n
        self.zp = nn.Linear(3 * d, d, bias=False)
        self.yp = nn.Linear(2 * d, d, bias=False)
    def forward(self, x, y, z):
        for _ in range(self.n):
            z = self.net(self.zp(torch.cat([x, y, z], dim=-1)))
        return self.net(self.yp(torch.cat([y, z], dim=-1))), z


class TRM(nn.Module):
    def __init__(self, d=256, l=2, h=4, n=4, T=4, v=5):
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
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ── Maze generation ───────────────────────────────────────────────────────────

def _bfs(maze):
    """BFS from (0,0) to (rows-1, cols-1). Returns path or []."""
    rows, cols = maze.shape
    goal = (rows - 1, cols - 1)
    queue = deque([(0, 0, [(0, 0)])])
    visited = {(0, 0)}
    while queue:
        r, c, path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc, path + [(nr, nc)]))
    return []


def _path_to_dirs(path):
    """BFS path → direction token sequence (0=UP 1=DOWN 2=LEFT 3=RIGHT 4=STOP)."""
    dirs = []
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        if r2 > r1:   dirs.append(1)  # DOWN
        elif r2 < r1: dirs.append(0)  # UP
        elif c2 > c1: dirs.append(3)  # RIGHT
        else:         dirs.append(2)  # LEFT
    dirs.append(4)  # STOP
    return dirs


def _generate_maze(size=10):
    """Generate a random solvable 10×10 maze, guaranteed to have a BFS path."""
    for _ in range(100):
        maze = np.zeros((size, size), dtype=np.int32)
        wall_density = random.uniform(0.20, 0.35)
        for i in range(size):
            for j in range(size):
                if (i, j) not in ((0, 0), (size - 1, size - 1)):
                    if random.random() < wall_density:
                        maze[i, j] = 1
        path = _bfs(maze)
        if len(path) >= 3:
            return maze, path
    # Fallback: corridor maze that's always solvable
    maze = np.zeros((size, size), dtype=np.int32)
    path = _bfs(maze)
    return maze, path


def generate_dataset(n_mazes, output_path, size=10):
    """Generate n_mazes and save to output_path as JSON."""
    data = []
    for i in range(n_mazes):
        maze, path = _generate_maze(size)
        dirs = _path_to_dirs(path)
        data.append({
            "maze": maze.flatten().tolist(),
            "directions": dirs,
        })
        # 8 dihedral augmentations per maze
        for k in [1, 2, 3]:
            aug = np.rot90(maze, k=k).copy()
            aug_path = _bfs(aug)
            if aug_path:
                data.append({
                    "maze": aug.flatten().tolist(),
                    "directions": _path_to_dirs(aug_path),
                })
        for fn in [np.fliplr, np.flipud]:
            aug = fn(maze).copy()
            aug_path = _bfs(aug)
            if aug_path:
                data.append({
                    "maze": aug.flatten().tolist(),
                    "directions": _path_to_dirs(aug_path),
                })
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{n_mazes} base mazes ({len(data)} total with augmentation)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)
    avg_len = sum(len(d["directions"]) for d in data) / len(data)
    print(f"Saved {len(data)} examples to {output_path}  (avg path length: {avg_len:.1f})")
    return data


# ── Dataset ───────────────────────────────────────────────────────────────────

class MazeDataset(Dataset):
    def __init__(self, data, max_len=128):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.zeros(self.max_len, dtype=torch.long)
        x[:100] = torch.tensor(item["maze"], dtype=torch.long)

        dirs = item["directions"]
        capped = dirs[:self.max_len]
        y_true = torch.zeros(self.max_len, dtype=torch.long)
        y_true[:len(capped)] = torch.tensor(capped, dtype=torch.long)

        y_init = torch.zeros(self.max_len, dtype=torch.long)
        return {
            "x_tokens": x,
            "y_init_tokens": y_init,
            "y_true": y_true,
            "path_len": len(capped),
        }


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = args.device
    use_amp = device == "cuda"

    # ── Data ─────────────────────────────────────────────────────────────────
    data_path = Path("data/maze/train.json")
    if data_path.exists():
        print(f"Loading existing dataset from {data_path}")
        with open(data_path) as f:
            data = json.load(f)
    else:
        print(f"Generating {args.n_mazes} mazes (+ augmentation)...")
        data = generate_dataset(args.n_mazes, data_path)

    dataset = MazeDataset(data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Dataset: {len(dataset)} examples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TRM(d=256, l=2, h=4, n=4, T=args.T, v=5).to(device)
    emb = nn.Embedding(10, 256).to(device)
    print(f"Model: {model.count_parameters():,} params  T={args.T}")

    optimizer = optim.AdamW(
        list(model.parameters()) + list(emb.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    warmup = args.warmup
    scheduler = LambdaLR(optimizer, lambda s: min(1.0, s / warmup) if warmup > 0 else 1.0)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ema_model = EMA(model)
    ema_emb = EMA(emb)

    # ── Loop ──────────────────────────────────────────────────────────────────
    print(f"\nTraining T={args.T} for {args.iters} iters on {device}")
    print("=" * 60)

    best_acc = 0.0
    data_iter = iter(loader)
    t0 = time.time()

    for step in range(args.iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x_tok = batch["x_tokens"].to(device)
        y_tok = batch["y_init_tokens"].to(device)
        y_true = batch["y_true"].to(device)
        path_lens = batch["path_len"]

        model.train()
        emb.train()

        for _ in range(args.n_sup):
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    x = emb(x_tok)
                    y = emb(y_tok)
                    z = torch.zeros(x_tok.size(0), 128, 256, device=device)
                    (_, _), y_hat = model(x, y, z)
                    loss = F.cross_entropy(
                        y_hat.view(-1, y_hat.size(-1)), y_true.view(-1), ignore_index=0
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(emb.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                x = emb(x_tok)
                y = emb(y_tok)
                z = torch.zeros(x_tok.size(0), 128, 256, device=device)
                (_, _), y_hat = model(x, y, z)
                loss = F.cross_entropy(
                    y_hat.view(-1, y_hat.size(-1)), y_true.view(-1), ignore_index=0
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(emb.parameters()), 1.0
                )
                optimizer.step()

            scheduler.step()
            ema_model.update()
            ema_emb.update()

            # Feed prediction back as next y_init (iterative refinement)
            with torch.no_grad():
                y_tok = y_hat.argmax(dim=-1)

        # ── Eval ──────────────────────────────────────────────────────────────
        if step % args.log_every == 0:
            model.eval()
            emb.eval()
            ema_model.apply()
            ema_emb.apply()
            with torch.no_grad():
                x = emb(x_tok)
                y = emb(batch["y_init_tokens"].to(device))
                z = torch.zeros(x_tok.size(0), 128, 256, device=device)
                (_, _), y_hat = model(x, y, z)
                preds = y_hat.argmax(dim=-1)
                correct = total = 0
                for i in range(len(path_lens)):
                    pl = path_lens[i].item()
                    correct += (preds[i, :pl] == y_true[i, :pl]).sum().item()
                    total += pl
                acc = correct / max(total, 1)
            ema_model.restore()
            ema_emb.restore()
            best_acc = max(best_acc, acc)
            elapsed = time.time() - t0
            print(f"Step {step:5d}/{args.iters} | loss={loss.item():.4f} | "
                  f"acc={acc:.3f} | best={best_acc:.3f} | {elapsed:.0f}s")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step > 0 and step % args.save_every == 0:
            _save(model, emb, best_acc, args, Path(args.output) / f"maze_step{step}.pt")

    # Final save — overwrites maze_final.pt used by demo
    ema_model.apply()
    ema_emb.apply()
    out = Path(args.output) / "maze_final.pt"
    _save(model, emb, best_acc, args, out)
    print(f"\nDone! Best acc={best_acc:.1%}  Saved -> {out}")


def _save(model, emb, acc, args, path):
    torch.save({
        "model": model.state_dict(),
        "emb": emb.state_dict(),
        "acc": acc,
        "config": {"T": args.T, "dim": 256, "n_latent": 4, "vocab_size": 5},
    }, path)
    print(f"  Saved {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train TRM maze model")
    p.add_argument("--T",          type=int,   default=4,       help="Recursion depth (default 4; try 6 for best accuracy)")
    p.add_argument("--iters",      type=int,   default=20000,   help="Training iterations")
    p.add_argument("--n-mazes",    type=int,   default=3000,    help="Base mazes to generate (x5 with augmentation)")
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--warmup",     type=int,   default=500,     help="LR warmup steps")
    p.add_argument("--n-sup",      type=int,   default=3,       help="Deep supervision steps per batch")
    p.add_argument("--log-every",  type=int,   default=100)
    p.add_argument("--save-every", type=int,   default=5000)
    p.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output",     type=str,   default="outputs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
