# TRM Product Backlog

**Product**: TRM Educational Gaming Platform
**Vision**: Ultra-efficient AI-powered STEM games that teach reasoning through recursive problem-solving

## Product Overview

| Game | STEM Focus | Target Age | Key Outcome |
|------|------------|------------|-------------|
| Sudoku Forge | Math Logic | 8-14 | +40% deduction skills |
| Maze Navigator | Algorithms | 10-16 | Search mastery |
| ARC Abstracter | Cognition | 12+ | +30% transfer learning |

---

## Epic 1: Sudoku Forge (Priority: High)

### US-001: Generate Extreme Sudoku Puzzles
**As a** student,
**I want** challenging Sudoku puzzles (20-25 hints),
**So that** I can develop advanced logical deduction skills.

**Acceptance Criteria:**
- [ ] Generate valid 9x9 Sudoku with unique solutions
- [ ] Difficulty: 20-25 hints (Extreme level)
- [ ] Support data augmentation via rule-preserving shuffles
- [ ] Generate 1000+ training puzzles

**Story Points:** 5

---

### US-002: Recursive Reasoning Model
**As a** system,
**I want** an AI that shows step-by-step reasoning traces,
**So that** students see constraint propagation, not just answers.

**Acceptance Criteria:**
- [ ] TinyNet architecture < 7M parameters
- [ ] Deep supervision with N=16 supervision steps
- [ ] Latent z-traces capture backtracking logic
- [ ] Target: 87% puzzle-level accuracy

**Story Points:** 13

---

### US-003: Visual Reasoning Display
**As a** student,
**I want** to see "z-webs" visualization of model reasoning,
**So that** I understand how backtracks lead to "aha!" moments.

**Acceptance Criteria:**
- [ ] Visualize constraint propagation per cell
- [ ] Highlight elimination reasoning
- [ ] Animate recursion cycles

**Story Points:** 8

---

### US-004: Adaptive Difficulty
**As a** student,
**I want** puzzles that adapt to my skill level,
**So that** I'm challenged but not frustrated.

**Acceptance Criteria:**
- [ ] ACT halting mechanism for adaptive computation
- [ ] Easier puzzles for beginners (30+ hints)
- [ ] Progressive difficulty unlocks

**Story Points:** 5

---

## Epic 2: Maze Navigator (Priority: Medium)

### US-005: Procedural Maze Generation
**As a** student,
**I want** algorithmically generated mazes,
**So that** each puzzle is unique.

**Acceptance Criteria:**
- [ ] Generate mazes of varying complexity
- [ ] Support DFS and BFS exploration paths
- [ ] Dead-ends teach decomposition

**Story Points:** 5

---

### US-006: Path Evolution Visualization
**As a** student,
**I want** to see my path evolve with heuristics,
**So that** I learn search algorithms intuitively.

**Acceptance Criteria:**
- [ ] Show DFS vs BFS comparison
- [ ] Animate pathfinding recursion
- [ ] Adaptive dead-ends challenge

**Story Points:** 8

---

## Epic 3: ARC Abstracter (Priority: Future)

### US-007: Pattern Grid Recognition
**As an** advanced student,
**I want** abstract pattern recognition challenges,
**So that** I develop novel rule emergence skills.

**Acceptance Criteria:**
- [ ] Load ARC-AGI-1 dataset (800 train, 400 test)
- [ ] Multi-abstraction z-traces
- [ ] 45% target accuracy

**Story Points:** 13

---

## Backlog Summary

| Epic | Stories | Total Points | Priority |
|------|---------|--------------|----------|
| Sudoku Forge | 4 | 31 | Sprint 1 |
| Maze Navigator | 2 | 13 | Sprint 2 |
| ARC Abstracter | 1 | 13 | Sprint 3+ |

---

*Transfer to MS Planner: Create one task per User Story, add acceptance criteria as checklist items.*
