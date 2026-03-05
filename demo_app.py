import time
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from collections import deque

# Page config
st.set_page_config(
    page_title="TRM Academy — AI Learning Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Reset & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #080e1c !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid #1e2d4a !important;
}
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* ── Sidebar ── */
.sidebar-brand {
    padding: 1.5rem 0 1rem 0;
    text-align: center;
}
.sidebar-logo {
    font-size: 2.8rem;
    display: block;
    margin-bottom: 0.3rem;
}
.sidebar-name {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.sidebar-tagline {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.2rem;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #1e2d4a;
    margin: 1rem 0;
}
.sidebar-section {
    font-size: 0.65rem;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 0.5rem 0 0.3rem 0;
}
.sidebar-stat {
    background: #131c30;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
}
.sidebar-stat-label { color: #64748b; }
.sidebar-stat-val { color: #a78bfa; font-weight: 600; }

/* ── Radio nav override ── */
[data-testid="stRadio"] label {
    background: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
    padding: 0.45rem 0.6rem !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
}
[data-testid="stRadio"] label:hover {
    background: #131c30 !important;
    color: #e2e8f0 !important;
}
[data-testid="stRadio"] [aria-checked="true"] + div label,
[data-testid="stRadio"] input:checked + div {
    background: linear-gradient(135deg, #312e81, #4c1d95) !important;
    color: #c4b5fd !important;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f1f3d 0%, #1a1040 50%, #0f1f3d 100%);
    border: 1px solid #1e2d4a;
    border-radius: 20px;
    padding: 3.5rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at center, rgba(99,102,241,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #818cf8;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #e2e8f0 30%, #818cf8 70%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #64748b;
    max-width: 600px;
    margin: 0 auto 2rem auto;
    line-height: 1.7;
}

/* ── Feature Cards ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    margin-bottom: 2rem;
}
.feature-card {
    background: #0d1828;
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.feature-card:hover {
    border-color: #4f46e5;
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(79,70,229,0.15);
}
.feature-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    border-radius: 16px 16px 0 0;
}
.feature-icon { font-size: 2rem; margin-bottom: 0.8rem; }
.feature-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.4rem;
}
.feature-desc { font-size: 0.85rem; color: #64748b; line-height: 1.5; }
.feature-tag {
    display: inline-block;
    background: rgba(99,102,241,0.1);
    color: #818cf8;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-top: 0.8rem;
    border: 1px solid rgba(99,102,241,0.2);
}

/* ── Stat Cards ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: #0d1828;
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.stat-card .s-label {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.stat-card .s-value {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-card .s-delta {
    font-size: 0.75rem;
    color: #34d399;
    margin-top: 0.2rem;
    font-weight: 500;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 1.5rem 0 1rem 0;
}
.section-header .sh-icon {
    font-size: 1.3rem;
    background: rgba(99,102,241,0.12);
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
}
.section-header .sh-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
}
.section-header .sh-badge {
    font-size: 0.7rem;
    background: rgba(99,102,241,0.12);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.25);
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-left: auto;
}

/* ── Learn Panel ── */
.learn-panel {
    background: #0d1828;
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.learn-panel .lp-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.learn-panel .lp-body {
    font-size: 0.88rem;
    color: #94a3b8;
    line-height: 1.7;
}
.learn-panel .lp-body strong { color: #c4b5fd; }

/* ── Concept Box ── */
.concept-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(167,139,250,0.06));
    border: 1px solid rgba(99,102,241,0.2);
    border-left: 3px solid #6366f1;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}
.concept-box .cb-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #818cf8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.concept-box .cb-body {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.6;
}

/* ── Objective Banner ── */
.objective-banner {
    background: linear-gradient(135deg, rgba(56,189,248,0.06), rgba(99,102,241,0.06));
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}
.objective-banner .ob-icon { font-size: 1.1rem; margin-top: 0.1rem; }
.objective-banner .ob-text { font-size: 0.85rem; color: #7dd3fc; line-height: 1.5; }
.objective-banner .ob-text strong { color: #38bdf8; }

/* ── Game Panel ── */
.game-panel {
    background: #0d1828;
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 1.5rem;
}

/* ── Buttons ── */
.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    padding: 0.65rem 1.2rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.2px;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(79,70,229,0.35) !important;
    background: linear-gradient(135deg, #5b52f0, #8b46f5) !important;
}
.stButton>button:active { transform: translateY(0) !important; }

/* ── Progress ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
    border-radius: 999px !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #0d1828 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 12px !important;
    padding: 0.8rem !important;
}
[data-testid="stMetricValue"] { color: #a78bfa !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 0.8rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
    font-size: 0.9rem !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #818cf8 !important;
    border-bottom-color: #6366f1 !important;
}

/* ── Code blocks ── */
.stCodeBlock { background: #0a1020 !important; border: 1px solid #1e2d4a !important; border-radius: 10px !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: rgba(99,102,241,0.07) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
}

/* ── Sudoku ── */
.sudoku-container { background: #0d1828; padding: 1.5rem; border-radius: 16px; border: 1px solid #1e2d4a; }
.sudoku-title { color: #e2e8f0; font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; }
.sudoku-grid { display: inline-block; border: 3px solid #334155; border-radius: 10px; overflow: hidden; box-shadow: 0 8px 30px rgba(0,0,0,0.4); }
.sudoku-grid table { border-collapse: collapse; margin: 0; padding: 0; }
.sudoku-grid td {
    width: 48px; height: 48px; text-align: center;
    font-size: 1.3rem; font-weight: 700; color: #e2e8f0;
    border: 1px solid #1e2d4a; background: #0d1828;
}
.sudoku-grid td.original { color: #c4b5fd; background: #131c30; }
.sudoku-grid td.solved { color: #34d399; background: rgba(52,211,153,0.06); animation: popIn 0.25s ease; }
.sudoku-grid td.right-border { border-right: 2px solid #334155; }
.sudoku-grid td.bottom-border { border-bottom: 2px solid #334155; }

/* ── Arch diagram ── */
.arch-box {
    background: #0d1828;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    position: relative;
}
.arch-box .ab-title { font-size: 0.75rem; color: #475569; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; }
.arch-box .ab-name { font-size: 0.95rem; font-weight: 700; color: #c4b5fd; }
.arch-box .ab-sub { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }
.arch-arrow { text-align: center; color: #4f46e5; font-size: 1.4rem; padding: 0.2rem 0; }

/* ── Animations ── */
@keyframes popIn {
    0%  { transform: scale(0.7); opacity: 0; }
    70% { transform: scale(1.1); }
    100%{ transform: scale(1);   opacity: 1; }
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeSlide 0.4s ease; }

/* ── Footer ── */
.platform-footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #1e2d4a;
    margin-top: 2rem;
    color: #334155;
    font-size: 0.8rem;
}
.platform-footer a { color: #4f46e5; text-decoration: none; }

/* ── 2048 Game ── */
.g2048-board {
    display: inline-grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    background: #0a1020;
    border: 2px solid #1e2d4a;
    border-radius: 16px;
    padding: 14px;
}
.g2048-tile {
    width: 88px; height: 88px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800;
    font-size: 1.6rem;
    transition: all 0.12s ease;
    animation: popIn 0.15s ease;
    user-select: none;
}
.g2048-tile.empty {
    background: #131c30;
    border: 1px solid #1e2d4a;
}
.g2048-score-box {
    background: #0d1828;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    text-align: center;
    min-width: 90px;
}
.g2048-score-box .sc-label {
    font-size: 0.65rem; font-weight: 700; color: #475569;
    text-transform: uppercase; letter-spacing: 1.5px;
}
.g2048-score-box .sc-val {
    font-size: 1.5rem; font-weight: 800; color: #a78bfa;
}
.hint-arrow {
    font-size: 2.5rem; text-align: center;
    animation: popIn 0.25s ease;
}
</style>
""", unsafe_allow_html=True)

# Model Architecture Classes
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
        self.h, self.d = h, d//h
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        attn = F.softmax(q @ k.transpose(-2,-1) / (self.d**0.5), dim=-1)
        return self.o((attn @ v).transpose(1,2).reshape(B, L, D))

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
        for layer in self.ls: x = layer(x)
        return self.n(x)

class LatentRec(nn.Module):
    def __init__(self, net, d=256, n=4):
        super().__init__()
        self.net, self.n = net, n
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

# Cache models
@st.cache_resource
def load_maze_model():
    try:
        checkpoint = torch.load('outputs/maze_final.pt', map_location='cpu')
        T = checkpoint.get('config', {}).get('T', 2)
        model = TRM(d=256, l=2, h=4, n=4, T=T, v=5)
        model.load_state_dict(checkpoint['model'])
        emb = nn.Embedding(10, 256)
        emb.load_state_dict(checkpoint['emb'])
        model.eval()
        emb.eval()
        return model, emb, checkpoint.get('acc', 0)
    except Exception as e:
        st.error(f"Error loading maze model: {e}")
        return None, None, 0

@st.cache_resource
def load_sudoku_model():
    try:
        checkpoint = torch.load('outputs/sudoku_final.pt', map_location='cpu')
        return checkpoint.get('best_acc', 0)
    except Exception as e:
        st.error(f"Error loading sudoku model: {e}")
        return 0

@st.cache_resource
def load_2048_model():
    try:
        ck = torch.load('outputs/2048_trm.pt', map_location='cpu')
        cfg = ck.get('config', {"d": 128, "l": 2, "h": 4, "n": 4, "T": 3, "v": 4})
        model = TRM(d=cfg["d"], l=cfg["l"], h=cfg["h"], n=cfg["n"], T=cfg["T"], v=cfg["v"])
        model.load_state_dict(ck['model'])
        emb = nn.Embedding(12, cfg["d"])
        emb.load_state_dict(ck['emb'])
        model.eval(); emb.eval()
        return model, emb, ck.get('acc', 0)
    except Exception:
        return None, None, 0

# Helper functions
def create_sample_maze():
    """Generate a random 10x10 maze with varied layouts"""
    import random
    
    # Choose random generation method for variety
    method = random.choice(['random_walls', 'recursive_division', 'cellular'])
    
    if method == 'random_walls':
        # Random walls with moderate density
        maze = np.zeros((10, 10), dtype=int)
        wall_density = random.uniform(0.25, 0.35)
        for i in range(10):
            for j in range(10):
                if (i, j) != (0, 0) and (i, j) != (9, 9):
                    if random.random() < wall_density:
                        maze[i, j] = 1
    
    elif method == 'recursive_division':
        # Recursive division algorithm
        maze = np.zeros((10, 10), dtype=int)
        
        def divide(x, y, w, h):
            if w < 2 or h < 2:
                return
            
            # Choose orientation  
            horizontal = random.choice([True, False]) if w == h else h > w
            
            if horizontal:
                # Divide horizontally
                wall_y = y + random.randint(0, max(0, h - 2))
                passage_x = x + random.randint(0, max(0, w - 1))
                for i in range(x, min(x + w, 10)):
                    if i != passage_x and 0 <= wall_y < 10:
                        maze[wall_y, i] = 1
                if wall_y > y:
                    divide(x, y, w, wall_y - y)
                if wall_y + 1 < y + h:
                    divide(x, wall_y + 1, w, y + h - wall_y - 1)
            else:
                # Divide vertically
                wall_x = x + random.randint(0, max(0, w - 2))
                passage_y = y + random.randint(0, max(0, h - 1))
                for i in range(y, min(y + h, 10)):
                    if i != passage_y and 0 <= wall_x < 10:
                        maze[i, wall_x] = 1
                if wall_x > x:
                    divide(x, y, wall_x - x, h)
                if wall_x + 1 < x + w:
                    divide(wall_x + 1, y, x + w - wall_x - 1, h)
        
        divide(0, 0, 10, 10)
    
    else:  # cellular automata
        # Start with random noise
        maze = np.random.choice([0, 1], size=(10, 10), p=[0.6, 0.4])
        
        # Run cellular automata for smoothing
        for _ in range(2):
            new_maze = maze.copy()
            for i in range(10):
                for j in range(10):
                    neighbors = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < 10 and 0 <= nj < 10 and (di != 0 or dj != 0):
                                neighbors += maze[ni, nj]
                    # Smoothing rules
                    if neighbors >= 5:
                        new_maze[i, j] = 1
                    elif neighbors <= 2:
                        new_maze[i, j] = 0
            maze = new_maze
    
    # Guarantee start and goal are clear
    maze[0, 0] = 0
    maze[9, 9] = 0
    
    # Ensure a valid path exists - if not, create one
    path = bfs_maze(maze)
    if len(path) < 2:
        # No path found, clear a simple route
        for i in range(10):
            maze[i, min(i, 9)] = 0
        maze[9, 9] = 0
    
    return maze

def bfs_maze(maze):
    """Find path in maze using BFS"""
    rows, cols = maze.shape
    start, goal = (0, 0), (rows-1, cols-1)
    queue, visited = deque([(start, [start])]), {start}
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return [start, goal]

def is_valid_sudoku(grid, row, col, num):
    """Check if number can be placed in Sudoku grid"""
    # Check row
    if num in grid[row]:
        return False
    # Check column
    if num in grid[:, col]:
        return False
    # Check 3x3 box
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    if num in grid[box_row:box_row+3, box_col:box_col+3]:
        return False
    return True

def solve_sudoku_steps(grid):
    """Solve Sudoku and return clean forward animation steps (no backtracking)."""
    original = grid.copy()
    working = grid.copy()

    def solve(g):
        for i in range(9):
            for j in range(9):
                if g[i, j] == 0:
                    for num in range(1, 10):
                        if is_valid_sudoku(g, i, j, num):
                            g[i, j] = num
                            if solve(g):
                                return True
                            g[i, j] = 0
                    return False
        return True

    if not solve(working):
        return []

    # Build one step per empty cell, filling in solution values row-by-row
    steps = []
    current = original.copy()
    for i in range(9):
        for j in range(9):
            if original[i, j] == 0:
                current = current.copy()
                current[i, j] = working[i, j]
                steps.append((i, j, int(working[i, j]), current))
    return steps

def visualize_maze(maze, path=None):
    """Create plotly visualization of maze"""
    fig = go.Figure()
    
    # Draw maze
    fig.add_trace(go.Heatmap(
        z=maze,
        colorscale=[[0, 'white'], [1, 'black']],
        showscale=False
    ))
    
    # Draw path if provided
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines+markers',
            line=dict(color='#667eea', width=4),
            marker=dict(size=10, color='#764ba2'),
            name='Path'
        ))
    
    fig.update_layout(
        width=500, height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

# ── Main App ─────────────────────────────────────────────────────────────────
def main():

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <span class="sidebar-logo">🧠</span>
            <div class="sidebar-name">TRM Academy</div>
            <div class="sidebar-tagline">AI Learning Lab</div>
        </div>
        <hr class="sidebar-divider">
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Navigate</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", [
            "🏠  Home",
            "🗺️  Maze Lab",
            "🎯  Sudoku Lab",
            "🎮  2048 Lab",
            "🏗️  Architecture",
            "📈  Results",
        ], label_visibility="collapsed")

        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">Platform Stats</div>', unsafe_allow_html=True)
        _2048_model, _2048_emb, _2048_acc = load_2048_model()
        _2048_acc_str = f"{_2048_acc:.0%}" if _2048_acc else "—"
        st.markdown(f"""
        <div class="sidebar-stat"><span class="sidebar-stat-label">Maze Model</span><span class="sidebar-stat-val">64.5%</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-label">Sudoku Model</span><span class="sidebar-stat-val">43.4%</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-label">2048 TRM</span><span class="sidebar-stat-val">{_2048_acc_str}</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-label">Parameters</span><span class="sidebar-stat-val">2M</span></div>
        <div class="sidebar-stat"><span class="sidebar-stat-label">Architecture</span><span class="sidebar-stat-val">TRM T=4</span></div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <hr class="sidebar-divider">
        <div style="text-align:center;font-size:0.7rem;color:#1e2d4a;padding-bottom:0.5rem;">
            TRM Academy &nbsp;·&nbsp; Powered by PyTorch
        </div>
        """, unsafe_allow_html=True)
    
    # ── Home Page ─────────────────────────────────────────────────────────────
    if page == "🏠  Home":
        # Hero
        st.markdown("""
        <div class="hero fade-in">
            <div class="hero-badge">AI · Reasoning · Education</div>
            <div class="hero-title">Learn AI Reasoning<br>Through Interactive Games</div>
            <div class="hero-sub">
                Explore how a <strong style="color:#c4b5fd">Tiny Recursive Model</strong> navigates mazes
                and solves Sudoku puzzles — step by step, with explanations at every move.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        st.markdown("""
        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">🗺️</div>
                <div class="feature-title">Maze Pathfinding</div>
                <div class="feature-desc">Watch TRM navigate a 10×10 maze by predicting one direction at a time — compare it with optimal BFS.</div>
                <span class="feature-tag">Intermediate</span>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Sudoku Solver</div>
                <div class="feature-desc">See a backtracking algorithm fill a Sudoku grid cell-by-cell with animated progress.</div>
                <span class="feature-tag">Advanced</span>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔁</div>
                <div class="feature-title">Recursive Reasoning</div>
                <div class="feature-desc">Understand how T recursion cycles let a 2M-parameter model iteratively refine its answers.</div>
                <span class="feature-tag">Core Concept</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats row
        st.markdown("""
        <div class="stat-row">
            <div class="stat-card">
                <div class="s-label">Maze Accuracy</div>
                <div class="s-value">64.5%</div>
                <div class="s-delta">Target exceeded</div>
            </div>
            <div class="stat-card">
                <div class="s-label">Model Size</div>
                <div class="s-value">2M</div>
                <div class="s-delta">Parameters</div>
            </div>
            <div class="stat-card">
                <div class="s-label">Recursion Depth</div>
                <div class="s-value">T = 4</div>
                <div class="s-delta">Cycles</div>
            </div>
            <div class="stat-card">
                <div class="s-label">Tasks Supported</div>
                <div class="s-value">2</div>
                <div class="s-delta">Maze + Sudoku</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # What is TRM
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.markdown("""
            <div class="section-header">
                <div class="sh-icon">🧠</div>
                <div class="sh-title">What is TRM?</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="learn-panel">
                <div class="lp-body">
                    <strong>TRM (Tiny Recursive Model)</strong> is a compact AI architecture that solves
                    complex reasoning tasks by reusing a small set of parameters across multiple
                    computation cycles — instead of stacking many separate layers.<br><br>
                    The key insight: <strong>depth through time, not through width.</strong> A 2-layer
                    transformer run T=4 times behaves like an 8-layer network, but shares weights
                    across cycles, drastically reducing parameter count.<br><br>
                    This platform lets you <em>see</em> that reasoning process in action.
                </div>
            </div>

            <div class="concept-box">
                <div class="cb-title">Core Principle</div>
                <div class="cb-body">
                    At each recursion cycle, the model updates a latent "reasoning memory" <strong>z</strong>
                    using the input <strong>x</strong>, current proposal <strong>y</strong>, and previous memory.
                    After T cycles, <strong>y</strong> has been iteratively refined into a high-quality answer.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div class="section-header">
                <div class="sh-icon">🎓</div>
                <div class="sh-title">Learning Path</div>
            </div>
            """, unsafe_allow_html=True)
            steps = [
                ("1", "Start at Maze Lab", "See TRM navigate step-by-step"),
                ("2", "Try Sudoku Lab", "Watch cell-by-cell solving"),
                ("3", "Read Architecture", "Understand how it works"),
                ("4", "Check Results", "Compare T=2 vs T=4"),
            ]
            for num, title, desc in steps:
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:0.8rem;margin-bottom:0.9rem;">
                    <div style="background:linear-gradient(135deg,#4f46e5,#7c3aed);color:white;
                                font-size:0.75rem;font-weight:700;min-width:26px;height:26px;
                                border-radius:50%;display:flex;align-items:center;justify-content:center;">
                        {num}
                    </div>
                    <div>
                        <div style="font-size:0.88rem;font-weight:600;color:#e2e8f0">{title}</div>
                        <div style="font-size:0.78rem;color:#64748b;margin-top:0.1rem">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Deliverables
        st.markdown("""
        <div class="section-header">
            <div class="sh-icon">📋</div>
            <div class="sh-title">Sprint Deliverables</div>
            <div class="sh-badge">Sprint 1 Complete</div>
        </div>
        """, unsafe_allow_html=True)
        deliverables = [
            "Product Backlog", "Sprint Backlog", "Architecture Document",
            "Functional Specification", "UI Design", "Test Cases",
            "Sprint Retrospective", "Presentation Outline", "Trained Models"
        ]
        cols = st.columns(3)
        for i, item in enumerate(deliverables):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:#0d1828;border:1px solid #1e2d4a;border-radius:8px;
                            padding:0.55rem 0.8rem;margin-bottom:0.5rem;display:flex;
                            align-items:center;gap:0.5rem;font-size:0.85rem;color:#94a3b8;">
                    <span style="color:#34d399;font-size:0.9rem;">✓</span> {item}
                </div>
                """, unsafe_allow_html=True)
    
    # ── Maze Lab ──────────────────────────────────────────────────────────────
    elif page == "🗺️  Maze Lab":
        st.markdown("""
        <div class="section-header fade-in">
            <div class="sh-icon">🗺️</div>
            <div class="sh-title">Maze Pathfinding Lab</div>
            <div class="sh-badge">Intermediate</div>
        </div>
        <div class="objective-banner">
            <div class="ob-icon">🎯</div>
            <div class="ob-text">
                <strong>Learning Objective:</strong> Understand how a Tiny Recursive Model predicts
                navigation directions through an unseen maze using wall-aware inference —
                and compare it to the guaranteed-optimal BFS algorithm.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        model, emb, acc = load_maze_model()
        
        if model is not None:
            # Helper function: wall-aware path decoding using full logit distribution
            def directions_to_path(maze, dir_logits):
                """At each step pick the highest-probability direction that is
                actually valid (in-bounds and not a wall), ignoring STOP tokens."""
                path = [(0, 0)]
                r, c = 0, 0
                rows, cols = maze.shape
                dir_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

                for step in range(len(dir_logits)):
                    if (r, c) == (rows - 1, cols - 1):
                        break
                    # Rank the 4 movement directions by predicted probability
                    probs = torch.softmax(dir_logits[step], dim=-1)
                    ranked = probs[:4].argsort(descending=True).tolist()
                    for d in ranked:
                        dr, dc = dir_map[d]
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
                            r, c = nr, nc
                            path.append((r, c))
                            break  # took the best valid step; move to next position

                return path
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### 🗺️ TRM-Guided Pathfinding")
                
                # Initialize session state for maze
                if 'maze_trm_path' not in st.session_state:
                    st.session_state.current_maze = create_sample_maze()
                    st.session_state.bfs_path = bfs_maze(st.session_state.current_maze)
                    st.session_state.maze_trm_path = None
                    st.session_state.is_trm_animating = False
                    st.session_state.trm_step = 0
                    st.session_state.display_bfs = True
                
                # Add controls
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    if st.button("🔄 New Maze", key="new_maze"):
                        st.session_state.current_maze = create_sample_maze()
                        st.session_state.bfs_path = bfs_maze(st.session_state.current_maze)
                        st.session_state.maze_trm_path = None
                        st.session_state.is_trm_animating = False
                        st.session_state.trm_step = 0
                        st.rerun()
                
                with col_b:
                    if st.button("🧠 TRM Solve", key="trm_solve", disabled=st.session_state.is_trm_animating):
                        with st.spinner("Running TRM inference..."):
                            # Prepare input
                            x = torch.zeros(1, 128, dtype=torch.long)
                            x[0, :100] = torch.tensor(st.session_state.current_maze.flatten(), dtype=torch.long)
                            y_init = torch.zeros(1, 128, dtype=torch.long)
                            
                            # Inference
                            with torch.no_grad():
                                x_emb = emb(x)
                                y_emb = emb(y_init)
                                z = torch.zeros(1, 128, 256)
                                (_, _), y_hat = model(x_emb, y_emb, z)
                                
                                # Use full logits for wall-aware direction selection
                                dir_logits = y_hat[0, :100]  # [100, 5]

                            # Convert to path using wall-aware top-k selection
                            st.session_state.maze_trm_path = directions_to_path(st.session_state.current_maze, dir_logits)
                            st.session_state.trm_predicted_dirs = dir_logits.argmax(dim=-1).tolist()
                            st.session_state.is_trm_animating = True
                            st.session_state.trm_step = 0
                            st.session_state.display_bfs = False
                            st.rerun()
                
                with col_c:
                    if st.button("📍 BFS Path", key="show_bfs"):
                        st.session_state.display_bfs = True
                        st.session_state.is_trm_animating = False
                        st.rerun()
                
                with col_d:
                    if st.button("⏹️ Stop", key="stop_maze"):
                        st.session_state.is_trm_animating = False
                        st.rerun()
                
                # Determine which path to show
                if st.session_state.display_bfs:
                    visible_path = st.session_state.bfs_path
                    path_type = "BFS"
                    path_color = "#10b981"  # Green
                elif st.session_state.maze_trm_path:
                    visible_path = st.session_state.maze_trm_path[:st.session_state.trm_step+1]
                    path_type = "TRM"
                    path_color = "#8b5cf6"  # Purple
                else:
                    visible_path = []
                    path_type = "None"
                    path_color = "#10b981"
                
                # Display progress
                if visible_path:
                    if st.session_state.display_bfs:
                        st.success(f"📍 Showing {path_type} optimal path ({len(visible_path)} steps)")
                    else:
                        progress = (st.session_state.trm_step + 1) / len(st.session_state.maze_trm_path) if st.session_state.maze_trm_path else 0
                        st.progress(progress, text=f"Step {st.session_state.trm_step + 1}/{len(st.session_state.maze_trm_path)}")
                        
                        if st.session_state.is_trm_animating:
                            pos = visible_path[-1]
                            st.info(f"🧠 TRM navigating to ({pos[0]}, {pos[1]})")
                        elif st.session_state.trm_step == len(st.session_state.maze_trm_path) - 1:
                            goal_reached = visible_path[-1] == (9, 9)
                            if goal_reached:
                                st.success("✅ TRM reached goal!")
                            else:
                                st.warning(f"⚠️ TRM stopped at ({visible_path[-1][0]}, {visible_path[-1][1]}) - needs more training!")
                
                # Enhanced maze visualization with HTML grid (like demo.html)
                def render_maze_grid(maze, path, animate_step=None):
                    """Render maze as HTML grid with colored cells like demo.html"""
                    html = """
                    <style>
                    .maze-grid-container {
                        display: inline-block;
                        background: #1e293b;
                        padding: 1rem;
                        border-radius: 1rem;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                    }
                    .maze-grid {
                        display: grid;
                        grid-template-columns: repeat(10, 40px);
                        gap: 0;
                        border: 2px solid #334155;
                        border-radius: 0.5rem;
                        overflow: hidden;
                    }
                    .maze-cell {
                        width: 40px;
                        height: 40px;
                        border: 0.5px solid rgba(51, 65, 85, 0.3);
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 1.2rem;
                        transition: all 0.2s;
                    }
                    .maze-cell.wall {
                        background: #334155;
                    }
                    .maze-cell.empty {
                        background: #0f172a;
                    }
                    .maze-cell.start {
                        background: #3b82f6;
                    }
                    .maze-cell.end {
                        background: #ef4444;
                    }
                    .maze-cell.visited {
                        background: #fbbf24;
                        animation: visitCell 0.3s;
                    }
                    .maze-cell.path {
                        background: #10b981;
                        animation: pathCell 0.3s;
                    }
                    .maze-cell.trm-path {
                        background: #8b5cf6;
                        animation: pathCell 0.3s;
                    }
                    .maze-cell.current {
                        background: #fbbf24;
                        border: 2px solid white;
                        animation: pulse 0.5s infinite;
                    }
                    @keyframes visitCell {
                        from { transform: scale(0); }
                        to { transform: scale(1); }
                    }
                    @keyframes pathCell {
                        0% { transform: scale(0); }
                        50% { transform: scale(1.3); }
                        100% { transform: scale(1); }
                    }
                    @keyframes pulse {
                        0%, 100% { transform: scale(1); }
                        50% { transform: scale(1.1); }
                    }
                    </style>
                    <div class="maze-grid-container">
                        <div class="maze-grid">
                    """
                    
                    # Render each cell
                    for i in range(10):
                        for j in range(10):
                            cell_class = []
                            cell_content = ""
                            
                            # Determine cell type
                            if i == 0 and j == 0:
                                cell_class.append("start")
                                cell_content = "⭐"
                            elif i == 9 and j == 9:
                                cell_class.append("end")
                                cell_content = "🎯"
                            elif maze[i, j] == 1:
                                cell_class.append("wall")
                            elif (i, j) in path:
                                if animate_step is not None and (i, j) == path[min(animate_step, len(path)-1)]:
                                    cell_class.append("current")
                                    cell_content = "🔵"
                                elif st.session_state.display_bfs:
                                    cell_class.append("path")
                                else:
                                    cell_class.append("trm-path")
                            else:
                                cell_class.append("empty")
                            
                            html += f'<div class="maze-cell {" ".join(cell_class)}">{cell_content}</div>'
                    
                    html += """
                        </div>
                    </div>
                    """
                    return html
                
                # Render the maze grid
                if st.session_state.is_trm_animating and visible_path:
                    maze_html = render_maze_grid(st.session_state.current_maze, visible_path, st.session_state.trm_step)
                else:
                    maze_html = render_maze_grid(st.session_state.current_maze, visible_path)
                
                st.markdown(maze_html, unsafe_allow_html=True)
                
                # Legend
                st.markdown("""
                <div style='display: flex; gap: 1.5rem; justify-content: center; margin-top: 1rem; flex-wrap: wrap;'>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                        <div style='width: 20px; height: 20px; background: #3b82f6; border-radius: 4px;'></div>
                        <span>Start</span>
                    </div>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                        <div style='width: 20px; height: 20px; background: #ef4444; border-radius: 4px;'></div>
                        <span>End</span>
                    </div>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                        <div style='width: 20px; height: 20px; background: #fbbf24; border-radius: 4px;'></div>
                        <span>Current</span>
                    </div>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                        <div style='width: 20px; height: 20px; background: #10b981; border-radius: 4px;'></div>
                        <span>BFS Path</span>
                    </div>
                    <div style='display: flex; align-items: center; gap: 0.5rem;'>
                        <div style='width: 20px; height: 20px; background: #8b5cf6; border-radius: 4px;'></div>
                        <span>TRM Path</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Auto-advance through TRM path
                if st.session_state.is_trm_animating and st.session_state.maze_trm_path:
                    if st.session_state.trm_step < len(st.session_state.maze_trm_path) - 1:
                        time.sleep(0.15)  # 150ms delay
                        st.session_state.trm_step += 1
                        st.rerun()
                    else:
                        st.session_state.is_trm_animating = False

            with col2:
                st.markdown("### 🎯 Model Performance")
                
                st.metric("Model Accuracy", f"{acc:.1%}")
                st.metric("Parameters", "2M")
                st.metric("Vocab Size", "5 (Directions)")
                
                # Path comparison
                if st.session_state.maze_trm_path and st.session_state.bfs_path:
                    st.markdown("### 📊 Path Comparison")
                    bfs_len = len(st.session_state.bfs_path)
                    trm_len = len(st.session_state.maze_trm_path)
                    goal_reached = st.session_state.maze_trm_path[-1] == (9, 9)
                    
                    col_x, col_y = st.columns(2)
                    with col_x:
                        st.metric("BFS Path", f"{bfs_len} steps")
                    with col_y:
                        if goal_reached:
                            extra = trm_len - bfs_len
                            st.metric("TRM Path", f"{trm_len} steps", delta=f"{extra:+d}")
                        else:
                            st.metric("TRM Path", f"{trm_len} steps", delta="Incomplete")
                    
                    if goal_reached:
                        efficiency = min(100, int((bfs_len / trm_len) * 100))
                        st.progress(efficiency / 100, text=f"Efficiency: {efficiency}%")
                
                # Show predicted directions
                if st.session_state.maze_trm_path:
                    st.markdown("### 🧭 TRM Predictions")
                    dir_names = ["⬆️ UP", "⬇️ DOWN", "⬅️ LEFT", "➡️ RIGHT", "🛑 STOP"]
                    dirs_text = []
                    for d in st.session_state.trm_predicted_dirs[:10]:
                        if d < len(dir_names):
                            dirs_text.append(dir_names[d])
                    st.code(", ".join(dirs_text))
                
                st.markdown("### 🧠 How TRM Works")
                st.info("""
                **Direction Prediction:**
                1. Encodes 10×10 maze
                2. Runs T=2 recursive cycles
                3. Predicts sequence of moves
                4. Outputs: ⬆️⬇️⬅️➡️🛑
                
                **vs BFS:**
                - BFS: Guaranteed optimal
                - TRM: Learned strategy
                - TRM can generalize!
                """)
        else:
            st.error("Maze model not found. Make sure `outputs/maze_final.pt` exists.")
    
    # Sudoku Demo Page
    elif page == "🎯 Sudoku Demo":
        st.markdown("## 🎯 Sudoku Solving Demo")
        
        sudoku_acc = load_sudoku_model()
        
        # Statistics Section
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="value">43.4%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Parameters</h3>
                <div class="value">14.2M</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Training</h3>
                <div class="value">15,000 steps</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>Config</h3>
                <div class="value">N_sup=4, dim=512</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sample Sudoku grid
        sample_grid = np.array([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9]
        ])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="sudoku-container">', unsafe_allow_html=True)
            st.markdown('<div class="sudoku-title">🎮 Automated Sudoku Solver</div>', unsafe_allow_html=True)
            
            # Initialize session state
            if 'solving_steps' not in st.session_state:
                st.session_state.solving_steps = None
                st.session_state.original_grid = sample_grid.copy()
                st.session_state.is_solving = False
                st.session_state.current_step = 0
            
            # Solve button with auto-play
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🚀 Solve Sudoku", key="solve_sudoku", disabled=st.session_state.is_solving):
                    st.session_state.solving_steps = solve_sudoku_steps(st.session_state.original_grid)
                    st.session_state.is_solving = True
                    st.session_state.current_step = 0
                    st.rerun()
            
            with col_b:
                if st.button("🔄 Reset", key="reset_sudoku"):
                    st.session_state.solving_steps = None
                    st.session_state.is_solving = False
                    st.session_state.current_step = 0
                    st.rerun()
            
            # Display current state
            if st.session_state.solving_steps and st.session_state.current_step < len(st.session_state.solving_steps):
                current_grid = st.session_state.solving_steps[st.session_state.current_step][3]
                step_info = st.session_state.solving_steps[st.session_state.current_step]
                
                # Progress bar
                progress = (st.session_state.current_step + 1) / len(st.session_state.solving_steps)
                st.progress(progress, text=f"Step {st.session_state.current_step + 1}/{len(st.session_state.solving_steps)}")
                
                if st.session_state.is_solving:
                    st.info(f"Placing {step_info[2]} at position ({step_info[0]+1}, {step_info[1]+1})")
                else:
                    st.success("✅ Solving complete!")
            else:
                current_grid = st.session_state.original_grid
            
            # Render Sudoku grid with HTML
            def render_sudoku_grid(grid):
                html = """
                <style>
                .sudoku-grid {
                    display: inline-block;
                    border: 3px solid #333;
                    background-color: white;
                    border-radius: 8px;
                    overflow: hidden;
                }
                .sudoku-grid table {
                    border-collapse: collapse;
                    margin: 0;
                    padding: 0;
                }
                .sudoku-grid td {
                    width: 50px;
                    height: 50px;
                    text-align: center;
                    font-size: 1.4rem;
                    font-weight: 600;
                    color: #333;
                    border: 1px solid #ccc;
                    background-color: white;
                    transition: all 0.3s ease;
                }
                .sudoku-grid td.original {
                    font-weight: 700;
                    color: #000;
                }
                .sudoku-grid td.solved {
                    color: #667eea;
                    background-color: #f0f4ff;
                }
                .sudoku-grid td.right-border {
                    border-right: 3px solid #333;
                }
                .sudoku-grid td.bottom-border {
                    border-bottom: 3px solid #333;
                }
                </style>
                <div class="sudoku-grid">
                <table>
                """
                
                for i in range(9):
                    html += "<tr>"
                    for j in range(9):
                        value = int(grid[i, j])
                        original_value = int(st.session_state.original_grid[i, j])
                        cell_content = str(value) if value != 0 else ""
                        
                        classes = []
                        if original_value != 0:
                            classes.append("original")
                        elif value != 0:
                            classes.append("solved")
                        
                        if (j + 1) % 3 == 0 and j < 8:
                            classes.append("right-border")
                        if (i + 1) % 3 == 0 and i < 8:
                            classes.append("bottom-border")
                        
                        class_str = f' class="{" ".join(classes)}"' if classes else ''
                        html += f"<td{class_str}>{cell_content}</td>"
                    html += "</tr>"
                
                html += """
                </table>
                </div>
                """
                return html
            
            st.markdown(render_sudoku_grid(current_grid), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Auto-advance through steps
            if st.session_state.is_solving and st.session_state.solving_steps:
                if st.session_state.current_step < len(st.session_state.solving_steps) - 1:
                    time.sleep(0.2)  # 200ms delay between steps
                    st.session_state.current_step += 1
                    st.rerun()
                else:
                    st.session_state.is_solving = False

        with col2:
            st.markdown("### 📊 Solver Info")
            if st.session_state.solving_steps:
                filled = np.count_nonzero(current_grid)
                st.metric("Cells Filled", f"{filled}/81")
                st.metric("Progress", f"{int(filled/81*100)}%")
                if st.session_state.is_solving:
                    st.metric("Status", "🔄 Solving...")
                else:
                    st.metric("Status", "✅ Complete")
            else:
                original_filled = np.count_nonzero(st.session_state.original_grid)
                st.metric("Given Clues", original_filled)
                st.metric("Empty Cells", 81 - original_filled)
            
            st.markdown("### 🧠 How it Works")
            st.info("""
            **Backtracking Algorithm:**
            1. Find empty cell
            2. Try digits 1-9
            3. Check if valid
            4. Recursively solve
            5. Backtrack if needed
            
            **TRM Approach:**
            - Encodes puzzle as tokens
            - Recursive refinement (T=2)
            - Outputs predictions
            """)
        
        st.markdown("")
        st.markdown("**Legend:** 🔵 Purple numbers are solved by the algorithm | Black numbers are original clues")
    
    # ── 2048 Lab ──────────────────────────────────────────────────────────────
    elif page == "🎮  2048 Lab":
        import random as _random
        import copy as _copy
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from src.data.game2048 import (
            move as _g_move, changed as _g_changed, add_tile as _g_add_tile,
            new_board as _g_new_board, game_over as _g_game_over, has_won as _g_has_won,
            valid_moves as _g_valid_moves, board_score as _g_board_score,
            random_move as _g_random_move, greedy_move as _g_greedy_move,
            expectimax_move as _g_expectimax_move, mcts_move as _g_mcts_move,
            trm_move as _g_trm_move,
        )

        # tile bg / text color
        _TC = {
            0:    ("#131c30", "#475569", "1px solid #1e2d4a"),
            2:    ("#1e3a5f", "#e2e8f0", "none"),
            4:    ("#1e4a7f", "#e2e8f0", "none"),
            8:    ("#2d6a9f", "#ffffff", "none"),
            16:   ("#3b82f6", "#ffffff", "none"),
            32:   ("#6366f1", "#ffffff", "none"),
            64:   ("#8b5cf6", "#ffffff", "none"),
            128:  ("#a78bfa", "#ffffff", "none"),
            256:  ("#c4b5fd", "#1e1b4b", "none"),
            512:  ("#34d399", "#ffffff", "none"),
            1024: ("#10b981", "#ffffff", "none"),
            2048: ("#f59e0b", "#000000", "none"),
        }

        _AI_NAMES  = ["Random", "Greedy", "Expectimax", "MCTS", "TRM"]
        _AI_COLORS = {"Random": "#64748b", "Greedy": "#3b82f6", "Expectimax": "#8b5cf6",
                      "MCTS": "#34d399", "TRM": "#f59e0b"}

        # ── helpers ────────────────────────────────────────────────────────
        def _render_board(board, cell_size="80px"):
            cells = ""
            for r in range(4):
                for c in range(4):
                    val = board[r][c]
                    bg, fg, border = _TC.get(val, ("#f59e0b", "#000", "none"))
                    fs = "1.5rem" if val < 100 else ("1.1rem" if val < 1000 else "0.9rem")
                    label = str(val) if val else ""
                    cells += (
                        f'<div style="background:{bg};border:{border};width:{cell_size};height:{cell_size};'
                        f'border-radius:8px;display:flex;align-items:center;justify-content:center;'
                        f'font-weight:800;font-size:{fs};color:{fg};">{label}</div>'
                    )
            return (
                '<div style="display:inline-grid;grid-template-columns:repeat(4,1fr);'
                f'gap:6px;background:#0a1020;border:2px solid #1e2d4a;border-radius:12px;padding:10px;">'
                f'{cells}</div>'
            )

        def _get_ai_move(ai_name, board, model, emb):
            if ai_name == "Random":     return _g_random_move(board)
            if ai_name == "Greedy":     return _g_greedy_move(board)
            if ai_name == "Expectimax": return _g_expectimax_move(board, depth=2)
            if ai_name == "MCTS":       return _g_mcts_move(board, n_simulations=30)
            if ai_name == "TRM":
                if model is None:
                    return _g_greedy_move(board)
                return _g_trm_move(board, model, emb)
            return _g_greedy_move(board)

        # ── session state: play tab ────────────────────────────────────────
        if "g2048_board" not in st.session_state:
            st.session_state.g2048_board = _g_new_board()
            st.session_state.g2048_score = 0
            st.session_state.g2048_best  = 0
            st.session_state.g2048_hint  = None

        # ── session state: race tab ────────────────────────────────────────
        if "race_running" not in st.session_state:
            st.session_state.race_running  = False
            st.session_state.race_boards   = {n: _g_new_board()  for n in _AI_NAMES}
            st.session_state.race_steps    = {n: 0               for n in _AI_NAMES}
            st.session_state.race_scores   = {n: 0               for n in _AI_NAMES}
            st.session_state.race_status   = {n: "playing"       for n in _AI_NAMES}
            st.session_state.race_max_tile = {n: 0               for n in _AI_NAMES}
            st.session_state.race_winner   = None

        def _race_reset():
            st.session_state.race_running  = False
            st.session_state.race_boards   = {n: _g_new_board()  for n in _AI_NAMES}
            st.session_state.race_steps    = {n: 0               for n in _AI_NAMES}
            st.session_state.race_scores   = {n: 0               for n in _AI_NAMES}
            st.session_state.race_status   = {n: "playing"       for n in _AI_NAMES}
            st.session_state.race_max_tile = {n: 0               for n in _AI_NAMES}
            st.session_state.race_winner   = None

        # ── page header ────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header fade-in">'
            '<div class="sh-icon">🎮</div>'
            '<div class="sh-title">2048 — TRM AI Strategy Lab</div>'
            '<div class="sh-badge">All Ages</div>'
            '</div>'
            '<div class="objective-banner">'
            '<div class="ob-icon">🎯</div>'
            '<div class="ob-text"><strong>Learning Objective:</strong> Compare 5 AI strategies — '
            'Random, Greedy, Expectimax, MCTS, and a trained <strong>TRM</strong> — '
            'racing to reach 2048 in the fewest moves.</div>'
            '</div>',
            unsafe_allow_html=True
        )

        tab_play, tab_race, tab_stats = st.tabs(["🎮 Play", "🏁 AI Race", "📊 Statistics"])

        # ══════════════════════════════════════════════════════════════════
        # Tab 1: Play
        # ══════════════════════════════════════════════════════════════════
        with tab_play:
            def _do_move(direction):
                nb, gained = _g_move(st.session_state.g2048_board, direction)
                if _g_changed(st.session_state.g2048_board, nb):
                    _g_add_tile(nb)
                    st.session_state.g2048_board = nb
                    st.session_state.g2048_score += gained
                    if st.session_state.g2048_score > st.session_state.g2048_best:
                        st.session_state.g2048_best = st.session_state.g2048_score
                st.session_state.g2048_hint = None

            board     = st.session_state.g2048_board
            game_over = _g_game_over(board)
            won       = _g_has_won(board)
            max_tile  = max(board[r][c] for r in range(4) for c in range(4))

            col_board, col_ctrl = st.columns([5, 3], gap="large")

            with col_board:
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Score",    st.session_state.g2048_score)
                sm2.metric("Best",     st.session_state.g2048_best)
                sm3.metric("Max Tile", max_tile)
                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                board_html = _render_board(board, "80px")
                if won:
                    board_html += (
                        '<div style="margin-top:10px;padding:8px 14px;background:rgba(251,191,36,0.1);'
                        'border:1px solid #f59e0b;border-radius:10px;color:#fbbf24;font-weight:700;'
                        'text-align:center;">🏆 You reached 2048! Keep going!</div>'
                    )
                if game_over:
                    board_html += (
                        '<div style="margin-top:10px;padding:8px 14px;background:rgba(239,68,68,0.1);'
                        'border:1px solid #ef4444;border-radius:10px;color:#f87171;font-weight:700;'
                        'text-align:center;">Game Over — No moves left!</div>'
                    )
                st.markdown(board_html, unsafe_allow_html=True)

            with col_ctrl:
                st.markdown(
                    '<div class="learn-panel" style="margin-bottom:12px;">'
                    '<div class="lp-title">🕹️ Move Controls</div>'
                    '</div>', unsafe_allow_html=True
                )
                _, mc, _ = st.columns([1, 2, 1])
                with mc:
                    if st.button("⬆️ Up", disabled=game_over, key="play_up"):
                        _do_move("up"); st.rerun()
                ml, mid, mr = st.columns(3)
                with ml:
                    if st.button("⬅️ Left", disabled=game_over, key="play_left"):
                        _do_move("left"); st.rerun()
                with mid:
                    st.markdown("<div style='height:38px'></div>", unsafe_allow_html=True)
                with mr:
                    if st.button("Right ➡️", disabled=game_over, key="play_right"):
                        _do_move("right"); st.rerun()
                _, mc2, _ = st.columns([1, 2, 1])
                with mc2:
                    if st.button("⬇️ Down", disabled=game_over, key="play_down"):
                        _do_move("down"); st.rerun()

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="learn-panel" style="margin-bottom:10px;">'
                    '<div class="lp-title">🤖 AI Assistant</div>'
                    '<div class="lp-body">Greedy 1-step lookahead: scores every next board '
                    'by <strong>empty cells + corner bonus + monotonicity</strong>.</div>'
                    '</div>', unsafe_allow_html=True
                )
                ca, cb = st.columns(2)
                with ca:
                    if st.button("💡 Hint", disabled=game_over, key="play_hint"):
                        st.session_state.g2048_hint = _g_greedy_move(board); st.rerun()
                with cb:
                    if st.button("▶️ AI Move", disabled=game_over, key="play_ai_move"):
                        h = _g_greedy_move(board)
                        if h: _do_move(h)
                        st.rerun()

                if st.session_state.g2048_hint:
                    icons = {"up": "⬆️", "down": "⬇️", "left": "⬅️", "right": "➡️"}
                    icon  = icons.get(st.session_state.g2048_hint, "")
                    st.markdown(
                        f'<div style="margin-top:10px;padding:10px;background:rgba(167,139,250,0.1);'
                        f'border:1px solid #a78bfa;border-radius:10px;text-align:center;font-size:1rem;">'
                        f'AI says: <strong style="color:#a78bfa;">'
                        f'{st.session_state.g2048_hint.upper()}</strong> {icon}</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                if st.button("🔄 New Game", key="play_new_game"):
                    st.session_state.g2048_board = _g_new_board()
                    st.session_state.g2048_score = 0
                    st.session_state.g2048_hint  = None
                    st.rerun()

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            ep1, ep2, ep3 = st.columns(3)
            with ep1:
                st.markdown(
                    '<div class="concept-box">'
                    '<div class="cb-title">🎯 How the Game Works</div>'
                    '<div class="cb-body">Slide all tiles in one direction. Same numbers '
                    '<strong>merge</strong> into their sum. A new tile (2 or 4) appears '
                    'after every move. Goal: reach the <strong>2048 tile</strong>!<br><br>'
                    '<em>Tip: keep the biggest number in a corner.</em></div>'
                    '</div>', unsafe_allow_html=True
                )
            with ep2:
                st.markdown(
                    '<div class="concept-box">'
                    '<div class="cb-title">🤖 Greedy Heuristic AI</div>'
                    '<div class="cb-body">Scores each possible board using three rules:<br>'
                    '• <strong>Empty cells</strong> — more space = more options<br>'
                    '• <strong>Corner rule</strong> — keep max tile in a corner<br>'
                    '• <strong>Monotonicity</strong> — tiles decrease from corner<br><br>'
                    'Picks the move with the <strong>highest score</strong>.</div>'
                    '</div>', unsafe_allow_html=True
                )
            with ep3:
                st.markdown(
                    '<div class="concept-box">'
                    '<div class="cb-title">📚 CS Concept: Heuristics</div>'
                    '<div class="cb-body">A <strong>heuristic</strong> is a fast, good-enough '
                    'rule when finding the perfect answer is too slow. Chess engines, GPS, '
                    'and spell-checkers all use heuristics.<br><br>'
                    '<em>See the Race tab to compare 5 AI strategies head-to-head!</em>'
                    '</div>'
                    '</div>', unsafe_allow_html=True
                )

        # ══════════════════════════════════════════════════════════════════
        # Tab 2: AI Race
        # ══════════════════════════════════════════════════════════════════
        with tab_race:
            _trm_model, _trm_emb, _trm_acc = load_2048_model()
            if _trm_model is None:
                st.warning("TRM model not found (outputs/2048_trm.pt). TRM will fall back to Greedy. Run `python scripts/train_2048.py` to train it.")

            rc1, rc2 = st.columns([2, 1])
            with rc1:
                if not st.session_state.race_running:
                    if st.button("🚀 Start Race", key="race_start"):
                        _race_reset()
                        st.session_state.race_running = True
                        st.rerun()
                else:
                    if st.button("⏹ Stop Race", key="race_stop"):
                        st.session_state.race_running = False
                        st.rerun()
            with rc2:
                if st.button("🔄 Reset Race", key="race_reset"):
                    _race_reset()
                    st.rerun()

            # Winner banner
            if st.session_state.race_winner:
                wname  = st.session_state.race_winner
                wcolor = _AI_COLORS.get(wname, "#f59e0b")
                wsteps = st.session_state.race_steps[wname]
                st.markdown(
                    f'<div style="margin:12px 0;padding:14px 20px;background:rgba(245,158,11,0.1);'
                    f'border:2px solid {wcolor};border-radius:14px;text-align:center;">'
                    f'<span style="font-size:1.4rem;font-weight:800;color:{wcolor};">🏆 {wname} reached 2048 first!</span>'
                    f'<span style="color:#94a3b8;font-size:0.9rem;margin-left:12px;">in {wsteps} steps</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Render 5 mini-boards: top row 3, bottom row 2 centered
            def _render_ai_card(name):
                board  = st.session_state.race_boards[name]
                steps  = st.session_state.race_steps[name]
                status = st.session_state.race_status[name]
                mxtile = st.session_state.race_max_tile[name]
                color  = _AI_COLORS.get(name, "#64748b")
                status_color = {"playing": "#38bdf8", "won": "#f59e0b", "lost": "#ef4444"}.get(status, "#64748b")
                status_icon  = {"playing": "▶", "won": "🏆", "lost": "✗"}.get(status, "")
                st.markdown(
                    f'<div style="background:#0d1828;border:1px solid {color};border-radius:12px;'
                    f'padding:10px;margin-bottom:8px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                    f'<span style="font-weight:700;font-size:0.95rem;color:{color};">{name}</span>'
                    f'<span style="font-size:0.78rem;color:{status_color};font-weight:600;">'
                    f'{status_icon} {status.upper()}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(_render_board(board, "52px"), unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Steps",    steps)
                m2.metric("Max Tile", mxtile)
                m3.metric("Status",   status.capitalize())

            top1, top2, top3 = st.columns(3)
            with top1: _render_ai_card(_AI_NAMES[0])
            with top2: _render_ai_card(_AI_NAMES[1])
            with top3: _render_ai_card(_AI_NAMES[2])

            _, bot1, bot2, _ = st.columns([1, 2, 2, 1])
            with bot1: _render_ai_card(_AI_NAMES[3])
            with bot2: _render_ai_card(_AI_NAMES[4])

            # Advance one step per rerun while running
            if st.session_state.race_running:
                any_playing = False
                for name in _AI_NAMES:
                    if st.session_state.race_status[name] != "playing":
                        continue
                    any_playing = True
                    b = st.session_state.race_boards[name]
                    if st.session_state.race_steps[name] >= 2000:
                        st.session_state.race_status[name] = "lost"
                        continue
                    d = _get_ai_move(name, b, _trm_model, _trm_emb)
                    if d is None:
                        st.session_state.race_status[name] = "lost"
                        continue
                    nb, gained = _g_move(b, d)
                    if _g_changed(b, nb):
                        _g_add_tile(nb)
                        st.session_state.race_boards[name]   = nb
                        st.session_state.race_steps[name]   += 1
                        st.session_state.race_scores[name]  += gained
                        mt = max(nb[r][c] for r in range(4) for c in range(4))
                        st.session_state.race_max_tile[name] = mt
                        if mt >= 2048:
                            st.session_state.race_status[name] = "won"
                            if st.session_state.race_winner is None:
                                st.session_state.race_winner = name
                        elif _g_game_over(nb):
                            st.session_state.race_status[name] = "lost"
                    else:
                        st.session_state.race_status[name] = "lost"

                if not any_playing:
                    st.session_state.race_running = False

                time.sleep(0.05)
                st.rerun()

        # ══════════════════════════════════════════════════════════════════
        # Tab 3: Statistics
        # ══════════════════════════════════════════════════════════════════
        with tab_stats:
            import pandas as _pd
            _trm_model_s, _trm_emb_s, _ = load_2048_model()

            def _run_stats(n_games, model, emb, max_steps=500):
                results = {name: {"wins": 0, "steps": [], "max_tiles": []} for name in _AI_NAMES}
                for _ in range(n_games):
                    for name in _AI_NAMES:
                        b = _g_new_board()
                        steps = 0
                        won = False
                        while steps < max_steps:
                            if _g_game_over(b):
                                break
                            d = _get_ai_move(name, b, model, emb)
                            if d is None:
                                break
                            nb, _ = _g_move(b, d)
                            if _g_changed(b, nb):
                                _g_add_tile(nb)
                                b = nb
                                steps += 1
                                if _g_has_won(b):
                                    won = True
                                    break
                            else:
                                break
                        mt = max(b[r][c] for r in range(4) for c in range(4))
                        results[name]["steps"].append(steps)
                        results[name]["max_tiles"].append(mt)
                        if won:
                            results[name]["wins"] += 1
                return results

            st.markdown("#### Run N Games per AI and compare performance")
            n_sel = st.selectbox("Games per AI", [5, 10, 20], index=0, key="stats_n")

            if st.button("▶ Run Statistics", key="stats_run"):
                with st.spinner(f"Running {n_sel} games × 5 AIs (max 500 steps each)…"):
                    res = _run_stats(n_sel, _trm_model_s, _trm_emb_s, max_steps=500)

                rows = []
                for name in _AI_NAMES:
                    steps_list = res[name]["steps"]
                    wins       = res[name]["wins"]
                    tiles_list = res[name]["max_tiles"]
                    rows.append({
                        "AI":            name,
                        "Success Rate":  f"{wins / n_sel:.0%}",
                        "Avg Steps":     round(sum(steps_list) / len(steps_list)),
                        "Min Steps":     min(steps_list) if wins else "—",
                        "Max Steps":     max(steps_list),
                        "Avg Max Tile":  round(sum(tiles_list) / len(tiles_list)),
                    })
                df = _pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Plotly chart
                import plotly.graph_objects as _pgo
                avg_steps = [r["Avg Steps"] for r in rows]
                min_steps = [r["Min Steps"] if isinstance(r["Min Steps"], int) else None for r in rows]
                names     = [r["AI"] for r in rows]
                colors    = [_AI_COLORS[n] for n in names]
                fig = _pgo.Figure()
                fig.add_trace(_pgo.Bar(
                    x=names, y=avg_steps, name="Avg Steps",
                    marker_color=colors, opacity=0.85,
                ))
                valid_min = [(n, v) for n, v in zip(names, min_steps) if v is not None]
                if valid_min:
                    fig.add_trace(_pgo.Scatter(
                        x=[n for n, _ in valid_min],
                        y=[v for _, v in valid_min],
                        mode="markers", name="Min Steps",
                        marker=dict(symbol="star", size=16, color="#fbbf24",
                                    line=dict(color="#000", width=1)),
                    ))
                fig.update_layout(
                    paper_bgcolor="#080e1c", plot_bgcolor="#0d1828",
                    font=dict(color="#e2e8f0", family="Inter"),
                    title=dict(text="Steps to Reach 2048 by AI Strategy",
                               font=dict(size=14, color="#c4b5fd")),
                    xaxis=dict(title="AI Strategy", gridcolor="#1e2d4a"),
                    yaxis=dict(title="Steps", gridcolor="#1e2d4a"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d4a"),
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Highlight most efficient
                winners_rows = [r for r in rows if isinstance(r["Min Steps"], int)]
                if winners_rows:
                    best = min(winners_rows, key=lambda r: r["Min Steps"])
                    st.markdown(
                        f'<div style="padding:14px 18px;background:rgba(245,158,11,0.08);'
                        f'border:1px solid #f59e0b;border-radius:12px;margin-top:8px;">'
                        f'<strong style="color:#fbbf24;">Most efficient:</strong> '
                        f'<span style="color:#e2e8f0;">{best["AI"]} reached 2048 in as few as '
                        f'<strong style="color:#f59e0b;">{best["Min Steps"]} steps</strong></span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            cb1, cb2, cb3 = st.columns(3)
            with cb1:
                st.markdown(
                    '<div class="concept-box">'
                    '<div class="cb-title">📏 Minimum Steps</div>'
                    '<div class="cb-body"><strong>Minimum Steps</strong> = the fewest moves in any '
                    'single run to place a 2048 tile. Building up from pairs of 2s requires '
                    '500+ merges theoretically. Lower = more efficient strategic planning.</div>'
                    '</div>', unsafe_allow_html=True
                )
            with cb2:
                st.markdown(
                    '<div class="concept-box">'
                    '<div class="cb-title">📊 Success Rate</div>'
                    '<div class="cb-body"><strong>Success Rate</strong> = fraction of games where '
                    'the AI reached 2048 within 500 moves. Random rarely succeeds; Expectimax '
                    'and MCTS have the highest rates due to deeper planning ahead.</div>'
                    '</div>', unsafe_allow_html=True
                )
            with cb3:
                st.markdown(
                    '<div class="concept-box">'
                    '<div class="cb-title">🔍 CS: Search Algorithms</div>'
                    '<div class="cb-body"><strong>Greedy</strong>: 1-step lookahead — fast, '
                    'imperfect. <strong>Expectimax</strong>: game-tree search with probability '
                    'nodes. <strong>MCTS</strong>: random rollouts to estimate move quality. '
                    '<strong>TRM</strong>: learned policy from game data.</div>'
                    '</div>', unsafe_allow_html=True
                )

    # Training Results Page
    elif page == "📈 Training Results":
        st.markdown("## 📈 Training Results & Analysis")
        
        tab1, tab2 = st.tabs(["Maze Training", "Sudoku Training"])
        
        with tab1:
            st.markdown("### Maze Model - Direction Prediction")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Accuracy", "64.5%", "+14.5% vs target")
                st.metric("Parameters", "2M")
                st.metric("Training Time", "~2 hours")
            
            with col2:
                st.metric("Dataset Size", "1000 mazes")
                st.metric("Vocab Size", "5 (directions)")
                st.metric("Batch Size", "64")
            
            st.markdown("#### Configuration")
            st.code("""
{
    "dim": 256,
    "n_layers": 2,
    "n_heads": 4,
    "n_latent": 4,
    "T_cycles": 2,
    "vocab_size": 5,  # UP, DOWN, LEFT, RIGHT, STOP
    "lr": 1e-3,
    "N_sup": 3
}
            """)
            
            st.success("✅ Exceeded target of 50-60% accuracy!")
        
        with tab2:
            st.markdown("### Sudoku Model - Digit Prediction")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Accuracy", "43.4%")
                st.metric("Parameters", "14.2M")
                st.metric("Training Time", "~8 hours")
            
            with col2:
                st.metric("Dataset Size", "50,000 puzzles")
                st.metric("Vocab Size", "10 (digits)")
                st.metric("Training Steps", "15,000")
            
            st.markdown("#### Analysis")
            st.info("""
            **Challenges:**
            - Deep supervision implementation needs refinement
            - Task complexity higher than maze
            - Longer training time required
            
            **Sprint 2 Focus:**
            - Fix deep supervision from paper
            - Increase training time
            - Target: 70%+ accuracy
            """)
    
    # Architecture Page
    else:  # Architecture
        st.markdown("## 🏗️ TRM Architecture")
        
        st.markdown("### Core Components")
        
        tab1, tab2, tab3 = st.tabs(["TinyNet", "Recursive Module", "Training"])
        
        with tab1:
            st.markdown("""
            #### TinyNet (2-Layer Transformer)
            
            **Components:**
            - **RMSNorm:** Efficient normalization without bias
            - **Rotary Embeddings (RoPE):** Position-aware attention
            - **SwiGLU Activation:** Gated linear units
            - **Multi-head Attention:** 4-8 heads depending on task
            
            **Parameters:**
            - Maze: 256 dim, 4 heads
            - Sudoku: 512 dim, 8 heads
            """)
            
            st.code("""
class TinyNet(nn.Module):
    def __init__(self, dim, n_layers=2, n_heads=4):
        self.layers = [Block(dim, n_heads) for _ in range(n_layers)]
        self.norm = RMSNorm(dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
            """, language="python")
        
        with tab2:
            st.markdown("""
            #### Latent Recursion
            
            **Algorithm:**
            1. Initialize latent state z
            2. For n iterations:
               - Update z using concatenated [x, y, z]
            3. Update output y using [y, z]
            
            **T-Cycle Warmup:**
            - T-1 cycles without gradients (warmup)
            - Final cycle with gradients (training)
            """)
            
            st.code("""
def forward(self, x, y, z):
    # T-1 warmup cycles (no gradients)
    with torch.no_grad():
        for _ in range(T - 1):
            y, z = latent_recursion(x, y, z)
    
    # Final cycle (with gradients)
    y, z = latent_recursion(x, y, z)
    return y_hat, halt_prob
            """, language="python")
        
        with tab3:
            st.markdown("""
            #### Training Strategy
            
            **Deep Supervision:**
            - N_sup refinement steps per sample
            - Progressive output improvement
            
            **EMA (Exponential Moving Average):**
            - Stabilizes training
            - Decay: 0.999
            
            **Optimization:**
            - AdamW optimizer
            - Learning rate warmup
            - Gradient clipping
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>TRM Demo | Sprint 1 Review | January 2026</p>
        <p>Models: outputs/maze_final.pt (64.5%) | outputs/sudoku_final.pt (43.4%)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
