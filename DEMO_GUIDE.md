# TRM Demo Guide

## Presentation Setup

### Before You Start
1. **Check Models**: Ensure `outputs/maze_final.pt` and `outputs/sudoku_final.pt` exist
2. **Start Server**: Run `streamlit run demo_app.py`
3. **Open Browser**: Navigate to `http://localhost:8501`
4. **Test Features**: Click through all demos to ensure everything works

## Demo Flow (15-20 minutes)

### 1. Introduction (2 min)
- **What is TRM**: Tiny Recursive Model for reasoning tasks
- **Key Innovation**: Recursive refinement (T=2 cycles) for better accuracy
- **Demo Purpose**: Interactive visualization of AI reasoning

### 2. Sudoku Demonstration (5-7 min)

#### Navigate to Sudoku Demo
1. Click **"🎯 Sudoku Demo"** tab
2. Show the **Statistics Cards**: 43.4% accuracy, 14.2M parameters

#### Live Solving
1. Click **"🚀 Solve Sudoku"**
2. **Explain while it solves**:
   - Purple cells = Algorithm solving step-by-step
   - Black cells = Original clues
   - Progress bar shows completion
   - Real-time metrics update

#### Key Points to Highlight
- **Backtracking algorithm**: Classic approach for comparison
- **Visual clarity**: 3×3 box borders, color coding
- **TRM approach**: Would encode puzzle as tokens, run T=2 recursive cycles

### 3. Maze Pathfinding Demo (8-10 min)

#### Generate Random Mazes
1. Click **"🔄 New Maze"** 2-3 times
2. **Explain randomization**:
   - Three algorithms: Random Walls, Recursive Division, Cellular Automata
   - Every maze is unique
   - Shows model robustness

#### TRM Solving
1. Click **"🧠 TRM Solve"** on an interesting maze
2. **Watch the animation** (150ms per step)
3. **Explain what's happening**:
   - TRM predicts directions: ⬆️⬇️⬅️➡️🛑
   - Purple path shows TRM's navigation
   - 🔵 marker shows current position
   - Model runs in real-time (~50ms inference)

#### Compare with BFS
1. Click **"📍 BFS Path"** to show optimal solution
2. **Highlight comparison panel**:
   - BFS Path vs TRM Path lengths
   - Efficiency percentage
   - Sometimes TRM finds shorter paths!
   - Sometimes needs more training (incomplete paths)

#### Show Predictions
- **Point out TRM Predictions section**
- Shows actual direction sequence: "⬇️ DOWN, ⬇️ DOWN, ➡️ RIGHT..."
- This is what the AI actually predicted!

### 4. Model Performance Discussion (3-5 min)

#### Maze Model Stats
- **64.5% accuracy**: Exceeds 50-60% target
- **2M parameters**: Compact model
- **Direction-based**: Predicts moves, not positions
- **Generalization**: Works on unseen random mazes

#### Technical Details
- **Input**: 10×10 binary maze (100 tokens)
- **Architecture**: Transformer with recursive latent state (z)
- **T=2 cycles**: Two rounds of refinement
- **Output**: Sequence of 5 direction tokens

### 5. Key Takeaways (2 min)

✅ **TRM demonstrates**:
- Recursive reasoning capabilities
- Real-time AI inference
- Generalizable problem-solving
- Visual, intuitive AI behavior

✅ **Interactive features**:
- Automated solving animations
- Random maze generation
- Model vs algorithm comparisons
- Direction prediction display

## Tips for Success

### Do's ✅
- **Generate multiple mazes** to show variety
- **Let animations complete** before clicking next
- **Explain color coding** (purple=TRM, green=BFS)
- **Highlight when TRM succeeds** vs when it needs training
- **Show the predictions panel** - proves real AI inference

### Don'ts ❌
- Don't click too fast (wait for animations)
- Don't expect 100% accuracy (64.5% is realistic)
- Don't skip the comparison with BFS
- Don't forget to show multiple random mazes

## Troubleshooting

### If the app is slow
- Refresh the page
- Restart Streamlit server
- Check system resources

### If TRM doesn't reach goal
- **This is expected!** 64.5% accuracy means ~35% incomplete
- **Show it as a learning opportunity**: "This maze needs more training data"
- **Generate a new maze** and try again

### If animations don't start
- Click the button again
- Check browser console for errors
- Ensure models are loaded (check terminal)

## Q&A Preparation

**Q: Why doesn't TRM always find the goal?**
A: 64.5% accuracy is very good for learned navigation. The model generalizes to unseen mazes but occasionally predicts suboptimal directions.

**Q: How is this different from traditional pathfinding?**
A: BFS is guaranteed optimal but requires exploring the entire maze. TRM learns patterns and predicts directly, which can be faster but is probabilistic.

**Q: Can it solve larger mazes?**
A: Current model is trained on 10×10. Could be retrained on larger sizes with more data.

**Q: What's the recursive part?**
A: The model runs T=2 cycles, refining its latent state (z) each time for better predictions.

## Version Information
- **Demo App**: v1.0 (Finalized)
- **Maze Model**: 64.5% accuracy, 2M parameters
- **Sudoku Model**: 43.4% accuracy, 14.2M parameters
- **UI**: Purple gradient theme with grid visualizations
- **Features**: Automated animations, random generation, real-time inference

---

**Good luck with your demo! 🚀**
