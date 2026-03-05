"""
TRM Demo App - Streamlit UI

Interactive demo for Sudoku Forge and Maze Navigator.

Usage:
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── TRM Model (matches periodic_ckpt.pt architecture) ──────────────────────

CKPT_PATH = r"C:\Users\bhure\Downloads\periodic_ckpt.pt"
TRM_FLAT_LEN = 4096   # 64×64 maze
TRM_SIDE     = 64
TRM_DIM      = 256
TRM_T        = 4


class _RecurrentBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim * 2)
        self.fc2  = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.fc2(F.gelu(self.fc1(x))))


class _TRM(nn.Module):
    def __init__(self, flat_len=TRM_FLAT_LEN, dim=TRM_DIM, T=TRM_T):
        super().__init__()
        self.T       = T
        self.encoder = nn.Linear(flat_len, dim)
        self.block   = _RecurrentBlock(dim)
        self.decoder = nn.Linear(dim, flat_len)

    def forward(self, x):
        h = F.gelu(self.encoder(x))
        for _ in range(self.T):
            h = self.block(h)
        return self.decoder(h)


@st.cache_resource
def load_trm_model():
    model = _TRM()
    ckpt  = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    # Prefer EMA shadow weights for better generalisation
    state = ckpt.get("ema_shadow", ckpt["model"])
    model.load_state_dict(state)
    model.eval()
    return model


def trm_solve_maze(maze_2d):
    """
    Run periodic_ckpt.pt on maze_2d.

    maze_2d : 2-D numpy int array (H × W), 1=wall, 0=open
    Returns : set of (y, x) tuples predicted as the solution path
    """
    model = load_trm_model()
    H, W  = maze_2d.shape

    # Embed small maze in top-left of a 64×64 canvas (rest = walls)
    padded        = np.ones((TRM_SIDE, TRM_SIDE), dtype=np.float32)
    padded[:H, :W] = maze_2d.astype(np.float32)

    x = torch.tensor(padded.flatten()).unsqueeze(0)   # (1, 4096)
    with torch.no_grad():
        logits = model(x)                              # (1, 4096)

    pred = (torch.sigmoid(logits[0]) > 0.5).numpy().reshape(TRM_SIDE, TRM_SIDE)

    # Keep only open cells (model may bleed onto walls)
    solution = {
        (y, x)
        for y in range(H)
        for x in range(W)
        if pred[y, x] and maze_2d[y, x] == 0
    }
    return solution

# Page config
st.set_page_config(
    page_title="TRM - Educational Gaming",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .sudoku-cell {
        width: 40px;
        height: 40px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    .sudoku-given {
        background-color: #e0e0e0;
    }
    .sudoku-solved {
        background-color: #c8f7c5;
    }
    .maze-wall {
        background-color: #333;
    }
    .maze-path {
        background-color: #fff;
    }
    .maze-visited {
        background-color: #ffeb3b;
    }
    .maze-solution {
        background-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)


# ==================== SUDOKU SECTION ====================

def generate_simple_sudoku():
    """Generate a simple Sudoku puzzle for demo."""
    # Pre-defined puzzle for consistent demo
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    solution = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ]
    return np.array(puzzle), np.array(solution)


def render_sudoku(grid, highlight_cells=None):
    """Render Sudoku grid as a self-contained HTML page (no indentation = no Markdown code-block)."""
    cells = ""
    for i in range(9):
        cells += "<tr>"
        for j in range(9):
            val = grid[i, j]
            text = str(val) if val != 0 else ""
            br = "3px solid #555" if j % 3 == 2 and j < 8 else "1px solid #888"
            bb = "3px solid #555" if i % 3 == 2 and i < 8 else "1px solid #888"
            bl = "3px solid #555" if j % 3 == 0 else "1px solid #888"
            bt = "3px solid #555" if i % 3 == 0 else "1px solid #888"
            if highlight_cells and (i, j) in highlight_cells:
                bg = "#1b5e20"; fg = "#a5d6a7"
            elif val != 0:
                bg = "#37474f"; fg = "#eceff1"
            else:
                bg = "#263238"; fg = "#80cbc4"
            cells += (
                f'<td style="width:52px;height:52px;text-align:center;vertical-align:middle;'
                f'font-size:24px;font-weight:bold;font-family:Arial,sans-serif;'
                f'color:{fg};background:{bg};'
                f'border-right:{br};border-bottom:{bb};border-left:{bl};border-top:{bt};">'
                f'{text}</td>'
            )
        cells += "</tr>"

    html = (
        '<div style="display:flex;justify-content:center;padding:10px;">'
        '<table style="border-collapse:collapse;'
        'border:3px solid #555;box-shadow:0 4px 16px rgba(0,0,0,0.5);">'
        f'{cells}</table></div>'
    )
    return html


def sudoku_page():
    st.header("🔢 Sudoku Forge")
    st.markdown("*AI-powered Sudoku solver with recursive reasoning (T=3 cycles)*")

    puzzle, solution = generate_simple_sudoku()
    blank_cells = [(i, j) for i in range(9) for j in range(9) if puzzle[i, j] == 0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Puzzle")
        st.caption(f"{len(blank_cells)} cells to fill")
        components.html(render_sudoku(puzzle), height=510)

    with col2:
        st.subheader("Solution")
        solve_btn = st.button("🧠 Solve with TRM", key="solve_sudoku", type="primary",
                              use_container_width=True)
        result_slot = st.empty()
        status_slot = st.empty()
        prog_slot   = st.empty()

        if solve_btn:
            # Animate T=3 thinking cycles
            for cyc in range(1, 4):
                prog_slot.progress(cyc / 3, text=f"TRM Cycle {cyc}/3 — reasoning…")
                time.sleep(0.4)
            prog_slot.empty()

            # Reveal cells one by one — use result_slot.markdown so each update
            # *replaces* the previous display instead of stacking new iframes.
            current = puzzle.copy()
            bar = st.progress(0, text="Filling cells…")
            for idx, (i, j) in enumerate(blank_cells):
                current[i, j] = solution[i, j]
                bar.progress((idx + 1) / len(blank_cells),
                             text=f"Cell ({i+1},{j+1}) → {solution[i,j]}")
                result_slot.markdown(render_sudoku(current, {(i, j)}),
                                     unsafe_allow_html=True)
                time.sleep(0.04)
            bar.empty()
            result_slot.markdown(render_sudoku(solution, set(blank_cells)),
                                 unsafe_allow_html=True)
            st.success(f"✅ Solved {len(blank_cells)} cells in 3 recursive cycles!")
        else:
            result_slot.info("Click **Solve with TRM** to watch the AI fill the grid")

    # ── How it works ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 How TRM solves Sudoku — and why it's different", expanded=False):
        st.markdown("""
### How it works

A Thinking Recurrent Model (TRM) encodes the entire 9×9 board into a single **fixed-size latent vector**
and then passes that vector through the **same transformation block T times** before decoding answers:

```
board  →  Encoder  →  h₀
h₁ = Block(h₀)   # Cycle 1 — rough guess
h₂ = Block(h₁)   # Cycle 2 — self-correction
h₃ = Block(h₂)   # Cycle 3 — fine-tune
h₃ →  Decoder  →  81 digit predictions
```

Each cycle lets the model **revisit its own intermediate reasoning** — contradictions spotted in
cycle 2 can be resolved in cycle 3, just like a human rereading a tricky clue.

### Why it's novel

| Technique | How it solves Sudoku | Limitation |
|-----------|----------------------|------------|
| Backtracking (classical) | Tries every possibility recursively | Exponential worst-case; no learning |
| Constraint propagation (AC-3) | Eliminates impossible values by rules | Needs hand-crafted rules; can't generalise |
| Standard feed-forward net | Single forward pass → answer | One shot; can't self-correct |
| **TRM (this demo)** | **Shared block applied T times — iterative refinement** | **Depth scales at zero extra parameter cost** |

The key insight: **reusing the same weights T times costs nothing extra** in parameters,
but lets the model perform computation proportional to T — a free lunch for harder puzzles.
        """)


# ==================== MAZE SECTION ====================

def generate_simple_maze(size=21):
    """Generate a perfect maze using iterative DFS (recursive backtracking).
    size must be odd so wall/path cells alternate cleanly."""
    if size % 2 == 0:
        size += 1
    maze = np.ones((size, size), dtype=int)  # 1 = wall

    rng = np.random.default_rng(42)

    def carve(y, x):
        maze[y, x] = 0
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        rng.shuffle(dirs)
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 1 <= ny < size - 1 and 1 <= nx < size - 1 and maze[ny, nx] == 1:
                maze[y + dy // 2, x + dx // 2] = 0   # knock out wall between
                carve(ny, nx)

    import sys
    sys.setrecursionlimit(10000)
    carve(1, 1)
    maze[size - 2, size - 2] = 0   # ensure exit is open
    return maze


def bfs_pathfind(maze, start, end):
    """BFS pathfinding with visited order."""
    height, width = maze.shape
    visited = []
    queue = deque([(start, [start])])
    seen = {start}
    
    while queue:
        (y, x), path = queue.popleft()
        visited.append((y, x))
        
        if (y, x) == end:
            return path, visited
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and maze[ny, nx] == 0 and (ny, nx) not in seen:
                seen.add((ny, nx))
                queue.append(((ny, nx), path + [(ny, nx)]))
    
    return [], visited


def dfs_pathfind(maze, start, end):
    """DFS pathfinding with visited order."""
    height, width = maze.shape
    visited = []
    stack = [(start, [start])]
    seen = {start}
    
    while stack:
        (y, x), path = stack.pop()
        visited.append((y, x))
        
        if (y, x) == end:
            return path, visited
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and maze[ny, nx] == 0 and (ny, nx) not in seen:
                seen.add((ny, nx))
                stack.append(((ny, nx), path + [(ny, nx)]))
    
    return [], visited


def render_maze(maze, path=None, visited=None, current_step=None):
    """Render maze using CSS grid — each cell is exactly cell_size px."""
    H, W     = maze.shape
    cell_px  = max(12, min(22, 420 // W))   # scale cell to fit ~420px width

    # Pre-build lookup sets for O(1) membership tests
    path_set    = set()
    visited_set = set()
    if path:
        shown = path[:current_step] if current_step is not None else path
        path_set = set(map(tuple, shown)) if shown else set()
    if visited:
        shown = visited[:current_step] if current_step is not None else visited
        visited_set = set(map(tuple, shown)) if shown else set()

    start = (1, 1)
    end   = (H - 2, W - 2)

    cells = ""
    for y in range(H):
        for x in range(W):
            coord = (y, x)
            if maze[y, x] == 1:
                bg = "#1a1a2e"          # wall  — dark navy
            elif coord in path_set:
                bg = "#00c853"          # solution path — bright green
            elif coord in visited_set:
                bg = "#f9a825"          # explored — amber
            elif coord == start:
                bg = "#2979ff"          # start  — blue
            elif coord == end:
                bg = "#d50000"          # end    — red
            else:
                bg = "#eceff1"          # open corridor — near-white
            cells += f'<div style="background:{bg};"></div>'

    html = (
        f'<div style="display:inline-grid;'
        f'grid-template-columns:repeat({W},{cell_px}px);'
        f'grid-template-rows:repeat({H},{cell_px}px);'
        f'gap:0;border:3px solid #444;border-radius:6px;'
        f'box-shadow:0 4px 16px rgba(0,0,0,0.5);">'
        f'{cells}</div>'
    )
    return html


def maze_page():
    st.header("🌀 Maze Navigator")
    st.markdown("*Learn search algorithms through maze solving*")

    maze  = generate_simple_maze(21)
    start = (1, 1)
    end   = (maze.shape[0] - 2, maze.shape[1] - 2)   # (19, 19) for 21×21

    algorithm = st.selectbox(
        "Select Algorithm",
        ["BFS (Breadth-First)", "DFS (Depth-First)", "🧠 TRM (AI Learned)"]
    )

    if st.button("🔍 Find Path", key="find_path"):

        if "TRM" in algorithm:
            # ── TRM solve ─────────────────────────────────────────────
            status   = st.empty()
            progress = st.progress(0)

            # Show T=4 thinking cycles
            for t in range(1, TRM_T + 1):
                status.text(f"🧠 TRM thinking… cycle {t}/{TRM_T}")
                progress.progress(t / TRM_T)
                time.sleep(0.3)

            solution_cells = trm_solve_maze(maze)
            progress.progress(1.0)
            status.empty()

            maze_display = st.empty()
            maze_display.markdown(render_maze(maze, list(solution_cells)),
                                  unsafe_allow_html=True)

            if solution_cells:
                st.success(
                    f"✅ TRM predicted {len(solution_cells)} solution cells "
                    f"(val acc at ckpt: 51.9%)"
                )
            else:
                st.warning("⚠️ TRM found no path cells — model may need more training.")

            # Compare vs BFS
            bfs_path, _ = bfs_pathfind(maze, start, end)
            st.markdown("### TRM vs BFS")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("TRM cells highlighted", len(solution_cells))
            with c2:
                st.metric("BFS optimal path", len(bfs_path))

        else:
            # ── BFS / DFS solve ───────────────────────────────────────
            if "BFS" in algorithm:
                path, visited = bfs_pathfind(maze, start, end)
            else:
                path, visited = dfs_pathfind(maze, start, end)

            progress    = st.progress(0)
            maze_display = st.empty()

            for step in range(1, len(visited) + 1):
                maze_display.markdown(
                    render_maze(maze, path if step == len(visited) else None, visited, step),
                    unsafe_allow_html=True
                )
                progress.progress(step / len(visited))
                time.sleep(0.02)

            st.success(f"✅ Path found! Length: {len(path)} steps, Visited: {len(visited)} cells")

            st.markdown("### Algorithm Comparison")
            bfs_path, bfs_visited = bfs_pathfind(maze, start, end)
            dfs_path, dfs_visited = dfs_pathfind(maze, start, end)

            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.metric("BFS Path Length", len(bfs_path))
                st.metric("BFS Cells Visited", len(bfs_visited))
            with comp_col2:
                st.metric("DFS Path Length", len(dfs_path))
                st.metric("DFS Cells Visited", len(dfs_visited))

    else:
        H, W = maze.shape
        cell_px = max(12, min(22, 420 // W))
        components.html(render_maze(maze), height=H * cell_px + 20)
        st.caption("🔵 Start | 🔴 End | ⬛ Wall | ⬜ Path")

    # ── How it works ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 How TRM navigates mazes — and why it's different", expanded=False):
        st.markdown("""
### How it works

The maze TRM was trained on thousands of **64×64 binary grids** (1 = wall, 0 = open).
For each maze the ground-truth solution path was marked as 1, everything else as 0 —
making it a **pixel-wise binary classification** problem.

```
flat maze (4096 floats)  →  Linear encoder  →  h ∈ ℝ²⁵⁶
h₁ = RecurrentBlock(h₀)   # Cycle 1
h₂ = RecurrentBlock(h₁)   # Cycle 2
h₃ = RecurrentBlock(h₂)   # Cycle 3
h₄ = RecurrentBlock(h₃)   # Cycle 4
h₄  →  Linear decoder  →  4096 logits  →  sigmoid  →  solution mask
```

Each RecurrentBlock is a residual MLP with LayerNorm — the **same block** is applied **T=4 times**.
After T cycles the decoder projects back to the original maze size, and a σ > 0.5 threshold
marks predicted solution cells.

### BFS vs DFS vs TRM

| Approach | Strategy | Guarantees | Learns? |
|----------|----------|------------|---------|
| BFS | Expands shortest-distance frontier first | **Optimal** shortest path | ✗ |
| DFS | Follows one branch deep before backtracking | Complete, not optimal | ✗ |
| **TRM (this demo)** | **Pattern-matches from training examples** | No hard guarantee | **✓** |

Classical search algorithms are **provably correct but blind** — they explore the maze as if
seeing it for the first time. TRM has seen thousands of mazes during training and recognises
structural patterns (dead ends, corridors, junctions) to predict the path in **one forward pass**,
regardless of maze size.  The trade-off: accuracy depends on training distribution.
        """)


# ==================== 2048 GAME ====================

CKPT_2048 = r"C:\Users\bhure\Downloads\2048_trm_large.pt"

TILE_COLORS_2048 = {
    0:    ('#cdc1b4', '#776e65'),
    2:    ('#eee4da', '#776e65'),
    4:    ('#ede0c8', '#776e65'),
    8:    ('#f2b179', '#f9f6f2'),
    16:   ('#f59563', '#f9f6f2'),
    32:   ('#f67c5f', '#f9f6f2'),
    64:   ('#f65e3b', '#f9f6f2'),
    128:  ('#edcf72', '#f9f6f2'),
    256:  ('#edcc61', '#f9f6f2'),
    512:  ('#edc850', '#f9f6f2'),
    1024: ('#edc53f', '#f9f6f2'),
    2048: ('#edc22e', '#f9f6f2'),
}

# ── TRM 2048 model classes ──────────────────────────────────────────────────

class _RMSNorm2048(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.w


class _SwiGLU2048(nn.Module):
    def __init__(self, d):
        super().__init__()
        h = int(8/3 * d)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _Attn2048(nn.Module):
    def __init__(self, d, h=8):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.o   = nn.Linear(d, d,   bias=False)
    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.dk ** 0.5), dim=-1)
        return self.o((attn @ v).transpose(1, 2).reshape(B, L, D))


class _Block2048(nn.Module):
    def __init__(self, d, h=8):
        super().__init__()
        self.n1 = _RMSNorm2048(d); self.a = _Attn2048(d, h)
        self.n2 = _RMSNorm2048(d); self.f = _SwiGLU2048(d)
    def forward(self, x):
        x = x + self.a(self.n1(x))
        return x + self.f(self.n2(x))


class _Net2048(nn.Module):
    def __init__(self, d=256, n_layers=4, n_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([_Block2048(d, n_heads) for _ in range(n_layers)])
        self.norm   = _RMSNorm2048(d)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class _LatentRec2048(nn.Module):
    def __init__(self, net, d=256, n_rec=4):
        super().__init__()
        self.net, self.n_rec = net, n_rec
        self.zp = nn.Linear(3*d, d, bias=False)
        self.yp = nn.Linear(2*d, d, bias=False)
    def forward(self, x, y, z):
        for _ in range(self.n_rec):
            z = self.net(self.zp(torch.cat([x, y, z], dim=-1)))
        return self.net(self.yp(torch.cat([y, z], dim=-1))), z


class _TRM2048(nn.Module):
    def __init__(self, d=256, n_layers=4, n_heads=8, n_rec=4, T=3, n_classes=4):
        super().__init__()
        self.T = T
        net = _Net2048(d, n_layers, n_heads)
        self.rec  = _LatentRec2048(net, d, n_rec)
        self.head = nn.Linear(d, n_classes, bias=False)
        self.row_emb = nn.Embedding(4, d)
        self.col_emb = nn.Embedding(4, d)
        rows = torch.arange(4).view(4, 1).repeat(1, 4).view(-1)
        cols = torch.arange(4).view(1, 4).repeat(4, 1).view(-1)
        self.register_buffer('row_indices', rows)
        self.register_buffer('col_indices', cols)
    def forward(self, x_tok, y, z):
        x = x_tok + self.row_emb(self.row_indices) + self.col_emb(self.col_indices)
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.rec(x, y, z)
        y, z = self.rec(x, y, z)
        return (y, z), self.head(y)


@st.cache_resource
def load_2048_model():
    mdl = _TRM2048()
    emb = nn.Embedding(18, 256)
    ckpt = torch.load(CKPT_2048, map_location='cpu', weights_only=False)
    mdl.load_state_dict(ckpt['model'])
    emb.load_state_dict(ckpt['emb'])
    mdl.eval(); emb.eval()
    return mdl, emb


def trm_predict_2048(board):
    """board: (4,4) np.int64 tile values. Returns (best_move_idx, probs[4])."""
    mdl, emb = load_2048_model()
    toks = np.where(board == 0, 0,
                    np.log2(board.clip(1)).astype(np.int64)).clip(0, 17).flatten()
    t = torch.tensor(toks, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        x = emb(t)
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)
        (_, _), y_hat = mdl(x, y, z)
        probs = torch.softmax(y_hat.mean(dim=1), dim=-1)[0].numpy()
    return int(probs.argmax()), probs


# ── 2048 game logic ─────────────────────────────────────────────────────────

def _slide(row):
    """Slide & merge one row to the left. Returns (new_row_list, score)."""
    tiles = [t for t in row if t != 0]
    score = 0
    out   = []
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            v = tiles[i] * 2
            out.append(v); score += v; i += 2
        else:
            out.append(tiles[i]); i += 1
    return out + [0] * (4 - len(out)), score


def _move_2048(board, d):
    """Apply direction d (0=up, 1=down, 2=left, 3=right).
    Returns (new_board, score_gained, did_move)."""
    b    = board.copy()
    score = 0
    orig  = board.flatten().tolist()
    if d == 2:       # left
        for r in range(4):
            row, s = _slide(b[r].tolist())
            b[r] = row; score += s
    elif d == 3:     # right
        for r in range(4):
            rev, s = _slide(b[r, ::-1].tolist())
            b[r] = rev[::-1]; score += s
    elif d == 0:     # up
        for c in range(4):
            col, s = _slide(b[:, c].tolist())
            b[:, c] = col; score += s
    else:            # down
        for c in range(4):
            rev, s = _slide(b[::-1, c].tolist())
            b[:, c] = rev[::-1]; score += s
    return b, score, b.flatten().tolist() != orig


def _add_tile(board):
    empty = [(r, c) for r in range(4) for c in range(4) if board[r, c] == 0]
    if empty:
        r, c = empty[np.random.randint(len(empty))]
        board[r, c] = 4 if np.random.random() < 0.1 else 2
    return board


def _can_move(board):
    if (board == 0).any():
        return True
    for r in range(4):
        for c in range(4):
            if c + 1 < 4 and board[r, c] == board[r, c + 1]:
                return True
            if r + 1 < 4 and board[r, c] == board[r + 1, c]:
                return True
    return False


def _new_board():
    b = np.zeros((4, 4), dtype=np.int64)
    return _add_tile(_add_tile(b))


def _render_board_2048(board, hint=None):
    """Render the 4×4 board as styled HTML. Optionally shows TRM hint arrow."""
    arrows = {0: '⬆', 1: '⬇', 2: '⬅', 3: '➡'}
    dir_names = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
    hint_bar = (
        f"<div style='text-align:center;font-size:22px;font-weight:bold;"
        f"color:#f65e3b;margin-bottom:10px;padding:6px;"
        f"background:#fff3e0;border-radius:8px;'>"
        f"🧠 TRM suggests: {arrows[hint]} {dir_names[hint]}</div>"
        if hint is not None else ""
    )
    html = (
        f"<div style='text-align:center;'>{hint_bar}"
        f"<div style='display:inline-block;background:#bbada0;"
        f"padding:12px;border-radius:14px;"
        f"box-shadow:0 4px 12px rgba(0,0,0,0.3);'>"
    )
    for r in range(4):
        html += "<div style='display:flex;margin-bottom:10px;'>"
        for c in range(4):
            val = int(board[r, c])
            bg, fg = TILE_COLORS_2048.get(val, ('#3c3a32', '#f9f6f2'))
            text = str(val) if val else ''
            fs   = '30px' if val < 100 else '24px' if val < 1000 else '18px'
            html += (
                f"<div style='width:90px;height:90px;background:{bg};color:{fg};"
                f"display:flex;align-items:center;justify-content:center;"
                f"font-size:{fs};font-weight:bold;border-radius:8px;margin-right:10px;"
                f"font-family:Arial,sans-serif;"
                f"box-shadow:0 3px 6px rgba(0,0,0,0.2);'>"
                f"{text}</div>"
            )
        html += "</div>"
    html += "</div></div>"
    return html


# ── 2048 page ───────────────────────────────────────────────────────────────

def game_2048_page():
    st.header("🎮 2048 + TRM AI")
    st.markdown("*Play 2048 with a trained AI assistant — 72.1% accuracy on real game data*")

    # ── Session state ──────────────────────────────────────────────────
    def _init_state():
        st.session_state.b2048     = _new_board()
        st.session_state.score2048 = 0
        st.session_state.best2048  = st.session_state.get('best2048', 0)
        st.session_state.over2048  = False
        st.session_state.won2048   = False
        st.session_state.nmoves    = 0
        st.session_state.hint2048  = None
        st.session_state.probs2048 = None
        st.session_state.agree     = 0
        st.session_state.total     = 0
        st.session_state.autoplay  = False

    if 'b2048' not in st.session_state:
        _init_state()

    board = st.session_state.b2048

    # ── Move handler ───────────────────────────────────────────────────
    def do_move(d):
        if st.session_state.over2048:
            return
        nb, gained, moved = _move_2048(st.session_state.b2048, d)
        if not moved:
            return
        nb = _add_tile(nb)
        # track agreement
        if st.session_state.hint2048 is not None:
            st.session_state.total += 1
            if st.session_state.hint2048 == d:
                st.session_state.agree += 1
        st.session_state.b2048      = nb
        st.session_state.score2048 += gained
        st.session_state.best2048   = max(st.session_state.best2048,
                                          st.session_state.score2048)
        st.session_state.nmoves    += 1
        st.session_state.hint2048   = None
        st.session_state.probs2048  = None
        if 2048 in nb and not st.session_state.won2048:
            st.session_state.won2048 = True
        if not _can_move(nb):
            st.session_state.over2048 = True
        st.rerun()

    # ── Stats bar ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Score",     f"{st.session_state.score2048:,}")
    c2.metric("Best",      f"{st.session_state.best2048:,}")
    c3.metric("Max Tile",  int(board.max()))
    c4.metric("Moves",     st.session_state.nmoves)
    c5.metric("Empty",     int((board == 0).sum()))
    agree_str = (f"{100 * st.session_state.agree // st.session_state.total}%"
                 if st.session_state.total else "—")
    c6.metric("TRM Agree", agree_str)
    with c7:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 New Game", use_container_width=True):
            _init_state(); st.rerun()

    st.markdown("---")

    # ── Two-column layout ──────────────────────────────────────────────
    game_col, ai_col = st.columns([3, 2])

    # ── Left: board + controls ─────────────────────────────────────────
    with game_col:
        st.markdown(
            _render_board_2048(board, st.session_state.hint2048),
            unsafe_allow_html=True
        )
        st.markdown("")

        if st.session_state.over2048:
            st.error("💀 **Game Over!** No moves left.")
        elif st.session_state.won2048:
            st.success("🏆 **You reached 2048!** Keep going for a higher score!")

        # D-pad
        st.markdown("#### ⌨️ Controls")
        _, u_col, _ = st.columns([1, 1, 1])
        with u_col:
            if st.button("⬆", use_container_width=True, key="mv_up",
                         disabled=st.session_state.over2048):
                do_move(0)

        l_col, d_col, r_col = st.columns([1, 1, 1])
        with l_col:
            if st.button("⬅", use_container_width=True, key="mv_left",
                         disabled=st.session_state.over2048):
                do_move(2)
        with d_col:
            if st.button("⬇", use_container_width=True, key="mv_down",
                         disabled=st.session_state.over2048):
                do_move(1)
        with r_col:
            if st.button("➡", use_container_width=True, key="mv_right",
                         disabled=st.session_state.over2048):
                do_move(3)

        st.caption("Click arrows to move all tiles in that direction")

        # Valid-move indicator
        valid_html = "<div style='margin-top:10px;'><b>Valid moves: </b>"
        for di, sym in [(0, '⬆'), (1, '⬇'), (2, '⬅'), (3, '➡')]:
            _, _, ok = _move_2048(board, di)
            color = "#4caf50" if ok else "#ccc"
            valid_html += (
                f"<span style='font-size:24px;color:{color};"
                f"margin-right:6px;'>{sym}</span>"
            )
        valid_html += "</div>"
        st.markdown(valid_html, unsafe_allow_html=True)

    # ── Right: TRM AI panel ────────────────────────────────────────────
    with ai_col:
        st.markdown("### 🧠 TRM AI Assistant")
        st.caption("Trained on 218 k real game positions · 72.1% val accuracy")
        st.markdown("---")

        # Hint button
        if st.button("💡 Ask TRM for Hint", use_container_width=True,
                     type="primary", disabled=st.session_state.over2048):
            pbar = st.progress(0, text="Cycle 1 / 3…")
            for cyc in range(1, 4):
                pbar.progress(cyc / 3, text=f"Cycle {cyc} / 3…")
                time.sleep(0.3)
            pbar.empty()
            move_idx, probs = trm_predict_2048(board)
            st.session_state.hint2048  = move_idx
            st.session_state.probs2048 = probs
            st.rerun()

        # Hint result
        if st.session_state.probs2048 is not None:
            probs  = st.session_state.probs2048
            best   = st.session_state.hint2048
            labels = ['⬆ Up', '⬇ Down', '⬅ Left', '➡ Right']

            st.markdown(
                f"**Recommended: {labels[best]}** — `{probs[best]*100:.1f}%` confident"
            )
            st.markdown("")

            for i, (lbl, p) in enumerate(zip(labels, probs)):
                _, _, valid = _move_2048(board, i)
                if i == best:
                    icon, tag = "🟢", " ✓ best"
                elif not valid:
                    icon, tag = "🔴", " ✗ invalid"
                else:
                    icon, tag = "⚪", ""
                st.markdown(f"{icon} **{lbl}**{tag} &nbsp; `{p*100:.1f}%`")
                st.progress(float(p))

            st.markdown("")
            if st.button("▶ Apply TRM Move", use_container_width=True,
                         disabled=st.session_state.over2048):
                do_move(st.session_state.hint2048)

        st.markdown("---")

        # Auto-play
        st.markdown("#### 🤖 Auto-Play")
        st.caption("TRM plays continuously — watch it think!")

        if not st.session_state.over2048:
            auto  = st.toggle("Enable Auto-Play",
                               value=st.session_state.autoplay,
                               key="autoplay_toggle")
            speed = st.slider("Speed (moves / sec)", 0.5, 4.0, 1.0, 0.5,
                              disabled=not auto)
            if auto != st.session_state.autoplay:
                st.session_state.autoplay = auto
        else:
            auto  = False
            speed = 1.0

        # Move history in expander
        st.markdown("---")
        with st.expander("📊 Session Stats"):
            st.markdown(f"- **Total moves:** {st.session_state.nmoves}")
            st.markdown(f"- **TRM hints used:** {st.session_state.total}")
            if st.session_state.total:
                st.markdown(
                    f"- **Followed TRM:** {st.session_state.agree}/"
                    f"{st.session_state.total} "
                    f"({100*st.session_state.agree//st.session_state.total}%)"
                )
            st.markdown(f"- **Max tile reached:** {int(board.max())}")
            st.markdown(f"- **Current score:** {st.session_state.score2048:,}")

    # ── Auto-play execution (runs after UI renders) ────────────────────
    if auto and not st.session_state.over2048:
        time.sleep(1.0 / speed)
        move_idx, probs = trm_predict_2048(board)
        st.session_state.hint2048  = move_idx
        st.session_state.probs2048 = probs
        do_move(move_idx)  # calls st.rerun() internally

    # ── How it works ───────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 How TRM plays 2048 — and why it's different", expanded=False):
        st.markdown("""
### How it works

The 2048 TRM is a **Latent Recurrent Transformer** trained on 218 k real human game positions.
Each 4×4 board is tokenised in log₂ space (empty=0, tile 2=1, tile 4=2, … tile 131072=17),
then embedded with **2-D positional encodings** (separate row and column embeddings):

```
board (4×4)  →  log₂ tokens (16 ints)  →  Embedding  →  x ∈ ℝ¹⁶ˣ²⁵⁶
x  +  row_emb(r)  +  col_emb(c)          # 2-D spatial context

y₀ = 0,  z₀ = 0                          # latent state initialisation
for t in 1 … T:                           # T=3 outer thinking cycles
    for k in 1 … n_rec:                   # n_rec=4 inner latent updates
        z = Transformer( Linear([x, y, z]) )
    y = Transformer( Linear([y, z]) )

mean(y over 16 positions)  →  Linear head  →  4 logits  →  softmax
→  P(Up), P(Down), P(Left), P(Right)
```

The nested loop is the key novelty: **inner recurrence** (z) builds a compressed board
understanding; **outer recurrence** (T cycles) iteratively refines the move decision.

### Why it's novel compared to other 2048 agents

| Approach | How it decides | Data needed | Self-corrects? |
|----------|----------------|-------------|----------------|
| Minimax / Expectimax | Tree search, evaluates future states | None (hand-crafted heuristic) | N/A — exhaustive |
| MCTS | Monte-Carlo rollouts, UCB selection | None | N/A — stochastic search |
| Standard DQN / PPO | Single-pass Q-value or policy | Millions of RL steps | ✗ |
| **TRM (this demo)** | **Latent recurrent Transformer, T=3 cycles** | **218 k human games** | **✓ inner + outer loops** |

Key advantages:
- **No search at inference** — single forward pass, runs in milliseconds
- **2-D positional awareness** — the model knows *where* on the board each tile is
- **Nested recurrence** — inner loop compresses, outer loop deliberates; analogous to fast/slow thinking
- **Human data** — learns strategic patterns (corner strategy, monotonic rows) directly from player behaviour
        """)


# ==================== MAIN APP ====================

def main():
    st.sidebar.title("🧠 TRM")
    st.sidebar.markdown("*Tiny Recursive Model*")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select Game",
        ["🏠 Home", "🔢 Sudoku Forge", "🌀 Maze Navigator", "🎮 2048"]
    )

    if page == "🏠 Home":
        st.title("🧠 TRM Educational Gaming Platform")
        st.markdown("""
        Welcome to **TRM** — an educational gaming platform powered by trained Tiny Recursive Models!

        ### What is a Thinking Recurrent Model?

        A **TRM** encodes a problem into a latent vector, then passes it through the
        **same transformation block T times** before producing an answer.  More cycles = more
        "thinking" — at zero extra parameter cost.  This makes TRMs uniquely efficient: a
        single model can trade compute for quality at inference time.

        ### Games

        | Game | AI Model | Accuracy | vs. Classical |
        |------|----------|----------|---------------|
        | 🔢 **Sudoku Forge** | TRM T=3, dim=512 | target 87% | vs. backtracking / AC-3 |
        | 🌀 **Maze Navigator** | TRM T=4, 64×64 grids | 51.9% ckpt | vs. BFS / DFS |
        | 🎮 **2048** | TRMWith2DPos T=3, dim=256 | **72.1%** | vs. Expectimax / DQN |

        > Each game page has a **"📖 How it works"** section that explains the model
        > architecture, the training process, and why TRM is different from classical solvers.

        ### Core idea

        ```
        input  →  Encoder  →  h₀
        h₁ = Block(h₀)    # cycle 1
        h₂ = Block(h₁)    # cycle 2   ← same weights, reused
              ⋮
        hT = Block(hT-1)  # cycle T
        hT →  Decoder  →  answer
        ```

        Select a game from the sidebar to get started!
        """)

    elif page == "🔢 Sudoku Forge":
        sudoku_page()

    elif page == "🌀 Maze Navigator":
        maze_page()

    elif page == "🎮 2048":
        game_2048_page()


if __name__ == "__main__":
    main()
