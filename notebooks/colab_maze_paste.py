# ========================================
# PASTE THIS DIRECTLY INTO GOOGLE COLAB
# ========================================

# Cell 1: Install
!pip install torch numpy opencv-python kaggle -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import time
import os
import glob
import cv2
from collections import deque

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Cell 2: Download Dataset
os.environ['KAGGLE_USERNAME'] = 'aayushbhure'
os.environ['KAGGLE_KEY'] = 'YOUR_KEY_HERE'  # CHANGE THIS

!kaggle datasets download -d emadehsan/rectangular-maze-kruskals-spanning-tree-algorithm
!unzip -o -q rectangular-maze-kruskals-spanning-tree-algorithm.zip
print("✓ Dataset ready")

# Cell 3: Model
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
        h = int(8/3*d)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Attn(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.h = h
        self.d = d // h
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.d ** 0.5), dim=-1)
        return self.o((attn @ v).transpose(1, 2).reshape(B, L, D))

class Block(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.a = Attn(d, h)
        self.n2 = RMSNorm(d)
        self.f = SwiGLU(d)
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
        self.net = net
        self.n = n
        self.zp = nn.Linear(3*d, d, bias=False)
        self.yp = nn.Linear(2*d, d, bias=False)
    def forward(self, x, y, z):
        for _ in range(self.n):
            z = self.net(self.zp(torch.cat([x, y, z], dim=-1)))
        return self.net(self.yp(torch.cat([y, z], dim=-1))), z

class TRM(nn.Module):
    def __init__(self, d=256, l=2, h=4, n=4, T=2, v=5):
        super().__init__()
        self.T = T
        net = Net(d, l, h)
        self.rec = LatentRec(net, d, n)
        self.head = nn.Linear(d, v, bias=False)
    def forward(self, x, y, z):
        with torch.no_grad():
            for _ in range(self.T-1):
                y, z = self.rec(x, y, z)
        y, z = self.rec(x, y, z)
        return (y, z), self.head(y)

print("✓ Model defined")

# Cell 4: Load Mazes
def img2maze(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, b = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
    return b

def bfs(m):
    r, c = m.shape
    s, g = (0, 0), (r-1, c-1)
    q, v = deque([(s, [s])]), {s}
    while q:
        (rr, cc), p = q.popleft()
        if (rr, cc) == g:
            return p
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = rr+dr, cc+dc
            if 0 <= nr < r and 0 <= nc < c and m[nr, nc] == 0 and (nr, nc) not in v:
                v.add((nr, nc))
                q.append(((nr, nc), p + [(nr, nc)]))
    return [s, g]

def path2dirs(p):
    if len(p) < 2:
        return [4]
    d = []
    for i in range(len(p)-1):
        r1, c1, r2, c2 = p[i][0], p[i][1], p[i+1][0], p[i+1][1]
        if r2 > r1:
            d.append(1)
        elif r2 < r1:
            d.append(0)
        elif c2 > c1:
            d.append(3)
        elif c2 < c1:
            d.append(2)
    d.append(4)
    return d

imgs = glob.glob('**/*.png', recursive=True)
print(f"Found {len(imgs)} images")

data = []
for p in imgs[:3000]:
    try:
        m = img2maze(p)
        if m is None or m.shape != (10, 10):
            continue
        path = bfs(m)
        if len(path) < 2:
            continue
        dirs = path2dirs(path[:30])
        data.append({"maze": m.flatten().tolist(), "dirs": dirs})
        if len(data) % 500 == 0:
            print(f"  {len(data)} mazes...")
    except:
        continue

print(f"\n✓ Loaded {len(data)} mazes")
if len(data) > 0:
    print(f"  Avg path: {np.mean([len(d['dirs']) for d in data]):.1f}")

# Cell 5: Dataset
class MazeData(Dataset):
    def __init__(self, d):
        self.d = d
    def __len__(self):
        return len(self.d)
    def __getitem__(self, i):
        it = self.d[i]
        x = torch.zeros(128, dtype=torch.long)
        x[:100] = torch.tensor(it["maze"], dtype=torch.long)
        y = torch.zeros(128, dtype=torch.long)
        yt = torch.zeros(128, dtype=torch.long)
        yt[:len(it["dirs"])] = torch.tensor(it["dirs"], dtype=torch.long)
        return {"x": x, "y": y, "yt": yt, "l": len(it["dirs"])}

ds = MazeData(data)
dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)
print(f"✓ Dataset ready: {len(ds)} samples")

# Cell 6: Setup Training
model = TRM(d=256, l=2, h=4, n=4, T=2, v=5).to(device)
emb = nn.Embedding(10, 256).to(device)
opt = optim.AdamW(list(model.parameters()) + list(emb.parameters()), lr=1e-3, weight_decay=0.01)
sch = LambdaLR(opt, lambda s: min(1.0, s/300))
sca = torch.amp.GradScaler('cuda')

class EMA:
    def __init__(self, m, d=0.999):
        self.m, self.d, self.s, self.b = m, d, {}, {}
        for n, p in m.named_parameters():
            if p.requires_grad:
                self.s[n] = p.data.clone()
    def update(self):
        for n, p in self.m.named_parameters():
            if p.requires_grad:
                self.s[n] = self.d * self.s[n] + (1 - self.d) * p.data
    def apply(self):
        for n, p in self.m.named_parameters():
            if p.requires_grad:
                self.b[n], p.data = p.data.clone(), self.s[n]
    def restore(self):
        for n, p in self.m.named_parameters():
            if p.requires_grad:
                p.data = self.b[n]

ema_m, ema_e = EMA(model), EMA(emb)
os.makedirs('/content/checkpoints', exist_ok=True)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Starting training...\n" + "="*60)

# Cell 7: Train
model.train()
it, t0, best = iter(dl), time.time(), 0.0

for s in range(6000):
    try:
        b = next(it)
    except StopIteration:
        it, b = iter(dl), next(iter(dl))
    
    x, yt, lens, y = b["x"].to(device), b["yt"].to(device), b["l"], b["y"].to(device)
    
    for _ in range(3):
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            ex, ey = emb(x), emb(y)
            z = torch.zeros(x.size(0), 128, 256, device=device)
            (_, _), yh = model(ex, ey, z)
            loss = F.cross_entropy(yh.view(-1, 5), yt.view(-1), ignore_index=0)
        sca.scale(loss).backward()
        sca.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(emb.parameters()), 1.0)
        sca.step(opt)
        sca.update()
        sch.step()
        ema_m.update()
        ema_e.update()
        with torch.no_grad():
            y = yh.argmax(dim=-1)
    
    if s % 50 == 0:
        with torch.no_grad():
            ema_m.apply()
            ema_e.apply()
            ex, ey, z = emb(x), emb(b["y"].to(device)), torch.zeros_like(ex)
            (_, _), yh = model(ex, ey, z)
            p = yh.argmax(dim=-1)
            c, tot = 0, 0
            for i in range(len(lens)):
                c += (p[i, :lens[i]] == yt[i, :lens[i]]).sum().item()
                tot += lens[i]
            acc = c / max(tot, 1)
            ema_m.restore()
            ema_e.restore()
        best = max(best, acc)
        print(f"Step {s:4d} | Loss: {loss.item():.4f} | Acc: {acc:.3f} | Best: {best:.3f} | {time.time()-t0:.0f}s")
    
    if s > 0 and s % 1000 == 0:
        ema_m.apply()
        ema_e.apply()
        torch.save({"model": model.state_dict(), "emb": emb.state_dict(), "acc": best},
                   f"/content/checkpoints/maze_step_{s}.pt")
        ema_m.restore()
        ema_e.restore()
        print(f"  ✓ Saved checkpoint")

ema_m.apply()
ema_e.apply()
torch.save({"model": model.state_dict(), "emb": emb.state_dict(), "acc": best},
           "/content/checkpoints/maze_final.pt")
print(f"\n✓ Complete! Best: {best:.1%}")
print(f"Checkpoints saved in /content/checkpoints/")
