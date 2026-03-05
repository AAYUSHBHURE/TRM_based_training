# Sprint 2 Backlog

**Sprint Goal**: Deliver Maze Navigator MVP and Streamlit Demo UI
**Duration**: 2 weeks
**Velocity Target**: 18-21 story points

---

## Committed User Stories

### US-005: Procedural Maze Generation (5 pts)
**Status**: ✅ Already Implemented

Implementation exists in `src/data/maze.py`:
- Recursive division algorithm
- 30x30 hard mazes with 110+ step paths
- 8 dihedral transforms for augmentation
- BFS pathfinding

### US-006: Path Evolution Visualization (8 pts)
**Status**: To Do

Display DFS vs BFS pathfinding animation.

### US-003: Visual Reasoning Display (8 pts)
**Status**: To Do

Show z-traces for Sudoku reasoning.

**Total Committed**: 21 points

---

## Task Breakdown

### US-005 Tasks (Already Done)

| Task | Status | Hours |
|------|--------|-------|
| T5.1: Recursive division generator | Done | 3 |
| T5.2: BFS pathfinding | Done | 2 |
| T5.3: Hard endpoint selection | Done | 2 |
| T5.4: Dihedral transforms | Done | 2 |
| T5.5: Dataset generation | Done | 1 |

---

### US-006 Tasks

| Task | Status | Hours |
|------|--------|-------|
| T6.1: Create Streamlit app structure | To Do | 2 |
| T6.2: Maze visualization component | To Do | 3 |
| T6.3: DFS pathfinding animation | To Do | 3 |
| T6.4: BFS pathfinding animation | To Do | 3 |
| T6.5: Algorithm comparison view | To Do | 2 |

---

### US-003 Tasks

| Task | Status | Hours |
|------|--------|-------|
| T3.1: Sudoku grid visualization | To Do | 2 |
| T3.2: Model inference integration | To Do | 3 |
| T3.3: z-trace extraction | To Do | 4 |
| T3.4: Reasoning step animation | To Do | 4 |

---

### Streamlit UI Tasks

| Task | Status | Hours |
|------|--------|-------|
| T-UI.1: App layout and navigation | To Do | 2 |
| T-UI.2: Sudoku Forge page | To Do | 3 |
| T-UI.3: Maze Navigator page | To Do | 3 |
| T-UI.4: Styling and polish | To Do | 2 |

---

## Definition of Done

- [ ] Streamlit app runs locally
- [ ] Sudoku demo shows model solving puzzles
- [ ] Maze demo shows pathfinding animation
- [ ] Both algorithms (DFS/BFS) visualized
- [ ] Documentation updated

---

*Transfer to MS Planner: Create Sprint 2 bucket, add tasks*
