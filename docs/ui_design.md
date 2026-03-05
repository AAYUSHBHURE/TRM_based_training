# TRM UI Design Document

## 1. Overview

**Module**: Sudoku Forge MVP
**Interface Type**: Command-Line Interface (CLI) + Python Scripts

> Note: Sprint 1 focuses on backend AI capabilities. Web/Mobile UI planned for Sprint 2+.

---

## 2. CLI Interface Design

### 2.1 Training CLI

```
┌─────────────────────────────────────────────────────────────┐
│  $ python -m src.train --config configs/sudoku_baseline.yaml│
├─────────────────────────────────────────────────────────────┤
│  ============================================================│
│  TRM Training                                                │
│  ============================================================│
│  Config: configs/sudoku_baseline.yaml                        │
│  Device: cuda                                                │
│  Model: 2 layers, dim=512                                    │
│  ============================================================│
│  Starting training for 50000 iterations                      │
│  Model parameters: 5,123,456                                 │
│                                                              │
│  Step 0/50000 | Loss: 2.3456 | Acc: 0.111 | Time: 0.1s      │
│  Step 100/50000 | Loss: 1.8234 | Acc: 0.234 | Time: 12.3s   │
│  Step 200/50000 | Loss: 1.4567 | Acc: 0.345 | Time: 24.5s   │
│  ...                                                         │
│  Saved: outputs/step_1000.pt                                 │
│  ...                                                         │
│  Training complete!                                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Evaluation CLI

```
┌─────────────────────────────────────────────────────────────┐
│  $ python -m src.evaluate --checkpoint outputs/final.pt     │
├─────────────────────────────────────────────────────────────┤
│  ============================================================│
│  TRM Sudoku Evaluation                                       │
│  ============================================================│
│  Checkpoint: outputs/final.pt                                │
│  Test puzzles: 100                                           │
│  Device: cuda                                                │
│  ============================================================│
│                                                              │
│  Evaluating puzzles...                                       │
│  [████████████████████████████████] 100/100                  │
│                                                              │
│  ═══════════════════════════════════════════════════════════│
│  RESULTS                                                     │
│  ═══════════════════════════════════════════════════════════│
│  Cell Accuracy:    92.3%                                     │
│  Puzzle Accuracy:  87.0%  ✓ Target met!                      │
│  Avg Inference:    45ms                                      │
│  ═══════════════════════════════════════════════════════════│
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Demo Visualization (Future Sprint)

### 3.1 Reasoning Trace Mockup

```
┌─────────────────────────────────────────────────────────────┐
│                    SUDOKU FORGE                              │
├───────────────────────┬─────────────────────────────────────┤
│                       │  Reasoning Trace                     │
│   5 3 · │ · 7 · │ · · ·  │                                     │
│   6 · · │ 1 9 5 │ · · ·  │  Cycle 1: Scanning constraints...   │
│   · 9 8 │ · · · │ · 6 ·  │  → R1C3: eliminated 5,3,7          │
│   ──────┼───────┼──────  │  → R1C3: candidates [4,6,8]        │
│   8 · · │ · 6 · │ · · 3  │                                     │
│   4 · · │ 8 · 3 │ · · 1  │  Cycle 2: Propagating...            │
│   7 · · │ · 2 · │ · · 6  │  → R1C3: forced = 4 ✓              │
│   ──────┼───────┼──────  │                                     │
│   · 6 · │ · · · │ 2 8 ·  │  Cycle 3: Finalizing...             │
│   · · · │ 4 1 9 │ · · 5  │  → Puzzle solved!                   │
│   · · · │ · 8 · │ · 7 9  │                                     │
│                       │  Confidence: 98.2%                   │
└───────────────────────┴─────────────────────────────────────┘
```

---

## 4. Workflow Diagram

```
User Flow (Sprint 1 MVP):

    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Generate │ ──► │  Train   │ ──► │ Evaluate │
    │  Data    │     │  Model   │     │  Results │
    └──────────┘     └──────────┘     └──────────┘
         │                │                │
         ▼                ▼                ▼
    train.json        final.pt        accuracy %
```

---

## 5. Future UI Roadmap

| Sprint | UI Feature |
|--------|------------|
| 2 | Streamlit web app |
| 3 | Reasoning trace visualization |
| 4 | Mobile-responsive design |
| 5 | Gamification elements |
