"""
game2048.py — Self-contained 2048 game logic, heuristics, and AI strategies.
Used by scripts/train_2048.py and demo_app.py.
"""
import random
import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset

DIRS = ["up", "left", "right", "down"]

# ── Token encoding ─────────────────────────────────────────────────────────────
# token 0 = empty, token k = log2(tile) for tile = 2^k (k=1..11 → 2..2048)
VOCAB_SIZE = 12  # tokens 0-11

def board_to_tokens(board):
    """Convert 4x4 board to LongTensor[16], row-major order."""
    tokens = []
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v == 0:
                tokens.append(0)
            else:
                tokens.append(int(math.log2(v)))
    return torch.tensor(tokens, dtype=torch.long)

# ── Game logic ─────────────────────────────────────────────────────────────────
def _merge_row(row):
    tiles = [x for x in row if x != 0]
    score, merged = 0, []
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged.append(tiles[i] * 2)
            score += tiles[i] * 2
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    while len(merged) < 4:
        merged.append(0)
    return merged, score

def move(board, direction):
    """Apply direction to board. Returns (new_board, score_gained)."""
    b = [row[:] for row in board]
    score = 0
    if direction == "left":
        for r in range(4):
            b[r], s = _merge_row(b[r]); score += s
    elif direction == "right":
        for r in range(4):
            rev, s = _merge_row(b[r][::-1]); b[r] = rev[::-1]; score += s
    elif direction == "up":
        for c in range(4):
            col = [b[r][c] for r in range(4)]
            merged, s = _merge_row(col); score += s
            for r in range(4): b[r][c] = merged[r]
    elif direction == "down":
        for c in range(4):
            col = [b[r][c] for r in range(4)]
            rev, s = _merge_row(col[::-1]); rev = rev[::-1]; score += s
            for r in range(4): b[r][c] = rev[r]
    return b, score

def changed(b1, b2):
    return any(b1[r][c] != b2[r][c] for r in range(4) for c in range(4))

def add_tile(board):
    empties = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
    if empties:
        r, c = random.choice(empties)
        board[r][c] = 4 if random.random() < 0.1 else 2
    return board

def new_board():
    b = [[0] * 4 for _ in range(4)]
    add_tile(b); add_tile(b)
    return b

def game_over(board):
    for r in range(4):
        for c in range(4):
            if board[r][c] == 0: return False
            if c + 1 < 4 and board[r][c] == board[r][c + 1]: return False
            if r + 1 < 4 and board[r][c] == board[r + 1][c]: return False
    return True

def has_won(board):
    return any(board[r][c] >= 2048 for r in range(4) for c in range(4))

def valid_moves(board):
    result = []
    for d in DIRS:
        nb, _ = move(board, d)
        if changed(board, nb):
            result.append(d)
    return result

# ── Heuristic ──────────────────────────────────────────────────────────────────
def board_score(board):
    """Combined heuristic: empty cells + corner bonus + monotonicity."""
    empties = sum(1 for r in range(4) for c in range(4) if board[r][c] == 0)
    max_tile = max(board[r][c] for r in range(4) for c in range(4))
    corners = [board[0][0], board[0][3], board[3][0], board[3][3]]
    corner_bonus = max_tile * 3 if max_tile in corners else 0
    mono = 0
    for r in range(4):
        for c in range(3):
            if board[r][c] >= board[r][c + 1]: mono += board[r][c + 1]
    for c in range(4):
        for r in range(3):
            if board[r][c] >= board[r + 1][c]: mono += board[r + 1][c]
    return empties * 150 + corner_bonus + mono

# ── AI strategies ──────────────────────────────────────────────────────────────
def random_move(board):
    """Pick a random valid direction."""
    moves = valid_moves(board)
    if not moves:
        return None
    return random.choice(moves)

def greedy_move(board):
    """1-step lookahead: pick move with highest heuristic score."""
    best_score, best_dir = -1e18, None
    for d in DIRS:
        nb, _ = move(board, d)
        if changed(board, nb):
            s = board_score(nb)
            if s > best_score:
                best_score, best_dir = s, d
    return best_dir

def _expectimax(board, depth, is_max_node):
    if depth == 0 or game_over(board):
        return board_score(board)
    if is_max_node:
        best = -1e18
        for d in DIRS:
            nb, _ = move(board, d)
            if changed(board, nb):
                val = _expectimax(nb, depth - 1, False)
                if val > best:
                    best = val
        return best if best > -1e18 else board_score(board)
    else:
        # Chance node: average over empty cells × {2:0.9, 4:0.1}
        empties = [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]
        if not empties:
            return _expectimax(board, depth - 1, True)
        total = 0.0
        for r, c in empties:
            for val, prob in [(2, 0.9), (4, 0.1)]:
                nb = [row[:] for row in board]
                nb[r][c] = val
                total += prob * _expectimax(nb, depth - 1, True)
        return total / len(empties)

def expectimax_move(board, depth=2):
    """Expectimax with chance nodes for tile placement."""
    best_score, best_dir = -1e18, None
    for d in DIRS:
        nb, _ = move(board, d)
        if changed(board, nb):
            val = _expectimax(nb, depth - 1, False)
            if val > best_score:
                best_score, best_dir = val, d
    return best_dir

def _greedy_rollout(board, n_steps=20):
    """Run a greedy rollout for MCTS simulation."""
    b = [row[:] for row in board]
    for _ in range(n_steps):
        if game_over(b):
            break
        d = greedy_move(b)
        if d is None:
            break
        nb, _ = move(b, d)
        if changed(b, nb):
            add_tile(nb)
            b = nb
    return board_score(b)

def mcts_move(board, n_simulations=30):
    """MCTS with greedy rollouts: pick direction with highest average rollout score."""
    best_score, best_dir = -1e18, None
    for d in DIRS:
        nb, _ = move(board, d)
        if changed(board, nb):
            add_tile(nb)
            total = sum(_greedy_rollout([row[:] for row in nb]) for _ in range(n_simulations))
            avg = total / n_simulations
            if avg > best_score:
                best_score, best_dir = avg, d
    return best_dir

def trm_move(board, model, emb):
    """TRM inference: encode board → embed → forward → argmax over valid moves."""
    tokens = board_to_tokens(board)               # [16]
    with torch.no_grad():
        x = emb(tokens.unsqueeze(0))              # [1, 16, d]
        y = torch.zeros_like(x)
        z = torch.zeros_like(x)
        (y_out, _), y_hat = model(x, y, z)        # y_hat: [1, 16, v]
        logits = y_hat[0].mean(dim=0)             # [v]  mean-pool over positions

    # Mask invalid moves
    moves = valid_moves(board)
    if not moves:
        return None

    # Mask directions not in valid_moves
    masked = logits.clone()
    for i, d in enumerate(DIRS):
        if d not in moves:
            masked[i] = float('-inf')

    best_idx = masked.argmax().item()
    if masked[best_idx] == float('-inf'):
        return random.choice(moves)
    return DIRS[best_idx]

# ── Training data generation ────────────────────────────────────────────────────
def generate_training_data(n_games=5000, depth=2):
    """
    Generate (board, move_label) pairs using Expectimax(depth) as teacher.
    Returns list of (board: list[list[int]], label: int)
    """
    data = []
    for game_idx in range(n_games):
        b = new_board()
        steps = 0
        while not game_over(b) and not has_won(b) and steps < 2000:
            d = expectimax_move(b, depth=depth)
            if d is None:
                break
            label = DIRS.index(d)
            data.append((copy.deepcopy(b), label))
            nb, _ = move(b, d)
            if changed(b, nb):
                add_tile(nb)
                b = nb
            steps += 1
        if (game_idx + 1) % 100 == 0:
            print(f"  Game {game_idx + 1}/{n_games} — {len(data)} samples")
    return data

# ── PyTorch Dataset ────────────────────────────────────────────────────────────
class Game2048Dataset(Dataset):
    def __init__(self, data):
        """data: list of (board, label) tuples."""
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board, label = self.samples[idx]
        return {
            "tokens": board_to_tokens(board),          # LongTensor[16]
            "label":  torch.tensor(label, dtype=torch.long),
        }
