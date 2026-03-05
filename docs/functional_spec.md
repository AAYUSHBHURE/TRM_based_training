# TRM Functional Specification

## 1. Overview

**Product**: TRM Educational Gaming Platform
**Module**: Sudoku Forge (Sprint 1 MVP)
**Version**: 1.0

---

## 2. Functional Requirements

### FR-001: Puzzle Generation
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-001.1 | Generate 9×9 Sudoku grids with unique solutions | High |
| FR-001.2 | Support Extreme difficulty (20-25 hints) | High |
| FR-001.3 | Apply rule-preserving augmentation | Medium |
| FR-001.4 | Export puzzles in JSON format | High |

### FR-002: AI Model Inference
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-002.1 | Load trained checkpoint (*.pt) | High |
| FR-002.2 | Accept puzzle as 81-token sequence | High |
| FR-002.3 | Return solution as 81-digit output | High |
| FR-002.4 | Apply recursive refinement (T=3 cycles) | High |
| FR-002.5 | Support batch inference for efficiency | Medium |

### FR-003: Evaluation
| ID | Requirement | Priority |
|----|-------------|----------|
| FR-003.1 | Calculate cell-level accuracy | High |
| FR-003.2 | Calculate puzzle-level accuracy | High |
| FR-003.3 | Report metrics to console | High |
| FR-003.4 | Save results to JSON | Low |

---

## 3. Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-001 | Model size | < 7M parameters |
| NFR-002 | Inference latency | < 100ms per puzzle |
| NFR-003 | Training time | < 4 hours (RTX 3070 Ti) |
| NFR-004 | Accuracy | ≥ 87% puzzle-level |

---

## 4. Input/Output Specifications

### Puzzle Input Format
```json
{
  "id": "sudoku_00001",
  "puzzle": [5,3,0,0,7,0,0,0,0,6,0,0,1,9,5,0,0,0,...],
  "solution": [5,3,4,6,7,8,9,1,2,6,7,2,1,9,5,3,4,8,...],
  "hints": 24
}
```

### Model Output Format
```
Input:  [B, 81] (token indices)
Output: [B, 81, 10] (logits for digits 0-9)
Halt:   [B, 1] (halting probability)
```

---

## 5. Workflow

```
1. Load Config        → sudoku_baseline.yaml
2. Load Checkpoint    → outputs/final.pt
3. Generate/Load Data → data/sudoku/train.json
4. Run Inference      → model.forward(x, y, z)
5. Decode Output      → argmax(y_hat)
6. Compare to Ground Truth
7. Report Accuracy
```

---

## 6. Error Handling

| Error | Handling |
|-------|----------|
| Invalid puzzle | Validate 0-9 values, 81 cells |
| No checkpoint | Exit with clear error message |
| CUDA OOM | Fall back to CPU |
| Invalid solution | Log as failed puzzle |
