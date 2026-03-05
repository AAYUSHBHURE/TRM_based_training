# 🧠 TRM: Tiny Recursive Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Model Accuracy](https://img.shields.io/badge/Maze%20Accuracy-64.5%25-green.svg)](outputs/maze_final.pt)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A comprehensive implementation and interactive demonstration of the **Tiny Recursive Model (TRM)** - a parameter-efficient transformer architecture that achieves deep reasoning through recursive computation. This project includes model training infrastructure, evaluation tools, and a production-ready Streamlit demo showcasing recursive reasoning on Sudoku and maze-solving tasks.

## 📚 Research Background

The Tiny Recursive Model represents a paradigm shift from **"Depth via Capacity"** to **"Depth via Time"**. Instead of relying on massive parameter counts and fixed-depth architectures, TRM leverages **algorithmic recurrence** - reusing a compact set of parameters iteratively to refine problem representations.

### Key Innovation
- **Recursive Refinement**: Single transformer block repeated T times (T=2 in our implementation)
- **Parameter Efficiency**: Achieves comparable performance to models 100x larger
- **Iterative Reasoning**: Mimics "System 2 thinking" through multiple refinement cycles
- **Deep Supervision**: Loss computed at each recursion step for stable training

For detailed theoretical background, see [`TRM_Concept_Explanation.txt`](TRM_Concept_Explanation.txt) and the [research paper](2510.04871v1.pdf).

## 🎯 Project Highlights

### ✅ Core Achievements
- 🏆 **64.5% accuracy** on 10×10 maze pathfinding (2M parameters)
- 🧩 **43.4% accuracy** on challenging Sudoku puzzles (14.2M parameters)
- 🚀 **Real-time inference** (~50ms per maze prediction)
- 🎨 **Production-ready demo** with modern UI and animations
- 📊 **Comprehensive training pipeline** with multiple task support

### 🎨 Interactive Demo Features
- **Maze Solving**: Real TRM inference with direction predictions (⬆️⬇️⬅️➡️🛑)
- **Sudoku Visualization**: Automated backtracking algorithm demonstration
- **3 Maze Generation Algorithms**: Random Walls, Recursive Division, Cellular Automata
- **BFS Comparison**: Side-by-side analysis of TRM vs optimal pathfinding
- **Animated Pathfinding**: Real-time visualization with progress tracking

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
CUDA 11.2+ (optional, for GPU training)
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd TRM

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Run the Interactive Demo
```bash
streamlit run demo_app.py
# Navigate to http://localhost:8501
```

### Train a Model
```bash
# Quick maze training example
python -m src.train --task maze --epochs 100 --batch_size 128

# See training notebooks for detailed examples
jupyter notebook notebooks/train_maze_10x10_FINAL.ipynb
```

## 📁 Complete Project Structure

```
TRM/
├── 📱 Demo Applications
│   ├── demo_app.py              # Main Streamlit application (46KB)
│   ├── app.py                   # Alternative demo interface
│   └── demo.html                # Original HTML visualization
│
├── 🧠 Source Code
│   └── src/
│       ├── __init__.py
│       ├── train.py             # Main training script
│       ├── evaluate.py          # Model evaluation utilities
│       ├── model/               # TRM architecture
│       │   ├── trm.py          # Core recursive model
│       │   ├── transformer.py   # Transformer components
│       │   └── embeddings.py   # Input/output embeddings
│       ├── data/                # Dataset loaders
│       │   ├── maze.py         # Maze generation & loading
│       │   ├── sudoku.py       # Sudoku dataset
│       │   └── utils.py        # Data utilities
│       ├── training/            # Training infrastructure
│       │   ├── trainer.py      # Training loop
│       │   ├── optimizer.py    # Optimization config
│       │   └── callbacks.py    # Training callbacks
│       └── utils/               # General utilities
│
├── 📓 Training Notebooks
│   ├── train_maze_10x10_FINAL.ipynb       # Final maze training (64.5%)
│   ├── train_colab_maze_directions.ipynb  # Direction-based approach
│   ├── TRM_Colab_A100.ipynb               # Sudoku training (A100 GPU)
│   ├── corrected_trm_training.py          # Bug-fixed training script
│   ├── deep_supervision_fix.py            # Deep supervision implementation
│   └── [15+ additional training experiments]
│
├── 🎯 Trained Models
│   └── outputs/
│       ├── maze_final.pt        # Maze model (7.6MB, 64.5% acc)
│       └── sudoku_final.pt      # Sudoku model (31MB, 43.4% acc)
│
├── 📖 Documentation
│   └── docs/
│       ├── PROJECT_SUMMARY.md           # Comprehensive project overview
│       ├── architecture.md              # System & model architecture
│       ├── DEMO_GUIDE.md                # Presentation instructions
│       ├── functional_spec.md           # Functional requirements
│       ├── ui_design.md                 # UI specifications
│       ├── test_cases.md                # Testing documentation
│       ├── DEPLOYMENT.md                # Deployment guide
│       ├── product_backlog.md           # Agile development tracking
│       ├── daily_scrum.md               # Development log
│       └── presentation_outline.md      # Demo presentation guide
│
├── 🧪 Testing & Validation
│   ├── tests/                   # Unit tests
│   ├── check_maze_model.py      # Maze model validator
│   └── check_sudoku_model.py    # Sudoku model validator
│
├── 🎨 Assets & Configuration
│   ├── assets/                  # Static resources
│   ├── configs/                 # Model configurations
│   ├── scenes/                  # Visualization scenes
│   └── web/                     # Web interface components
│
├── 📄 Project Files
│   ├── README.md                # This file
│   ├── DEMO_GUIDE.md            # Quick demo instructions
│   ├── requirements.txt         # Python dependencies
│   ├── .gitignore              # Git ignore rules
│   ├── 2510.04871v1.pdf        # Research paper
│   ├── TRM_Concept_Explanation.txt  # Deep technical dive
│   └── trm_paper_extracted.txt # Paper text extraction
│
└── 📊 Data & Outputs
    ├── data/                    # Training datasets
    ├── outputs/                 # Model checkpoints
    └── build/                   # Build artifacts
```

## 🎮 Demo Usage Guide

### 🧩 Sudoku Demonstration
1. Navigate to **"🎯 Sudoku Demo"** tab
2. View model statistics (43.4% accuracy, 14.2M parameters)
3. Click **"🚀 Solve Sudoku"** for automated solving
4. Watch step-by-step backtracking algorithm with color-coded cells:
   - **Black**: Original clues (bold)
   - **Purple**: Algorithm-solved cells
5. Track progress with cells filled counter
6. Click **"🔄 Reset"** to start fresh

### 🗺️ Maze Pathfinding Demo
1. Navigate to **"🗺️ Maze Demo"** tab
2. Click **"🔄 New Maze"** to generate random layout
   - Rotates between 3 algorithms for variety
3. Click **"🧠 TRM Solve"** to run AI inference
   - Watch real-time direction predictions
   - Purple path shows TRM navigation
   - 🔵 marker tracks current position
4. Click **"📍 BFS Path"** to show optimal solution
   - Green path displays shortest route
5. Compare efficiency metrics:
   - Path lengths (TRM vs BFS)
   - Success rate
   - Prediction sequence (⬆️⬇️⬅️➡️🛑)

## 🔬 Model Architecture

### TRM Core Design
```
Input → Embedding → [Recursive Block] × T → Projection → Output
                           ↑______|
```

**Recursive Block** (shared weights):
- RMSNorm / LayerNorm
- Multi-Head Self-Attention (MHSA)
- Feed-Forward Network (FFN)
- Residual connections

### Maze Model Specifications
- **Task**: 10×10 grid pathfinding
- **Input**: 100 tokens (binary maze: wall/path)
- **Vocabulary**: 5 directions (UP/DOWN/LEFT/RIGHT/STOP)
- **Architecture**: Recursive transformer, T=2 cycles
- **Parameters**: 2M
- **Accuracy**: 64.5%
- **Training**: Deep supervision at each recursion step

### Sudoku Model Specifications
- **Task**: 9×9 Sudoku solving
- **Input**: 81 tokens (digits 0-9)
- **Vocabulary**: 10 (digits)
- **Architecture**: Recursive transformer, T=2 cycles
- **Parameters**: 14.2M
- **Accuracy**: 43.4%
- **Training**: Full deep supervision (N_sup=16)

## 📊 Performance Metrics

| Task | Model | Accuracy | Parameters | Inference | Vocabulary | Grid Size |
|------|-------|----------|------------|-----------|------------|-----------|
| Maze Pathfinding | TRM | **64.5%** | 2M | ~50ms | 5 (dirs) | 10×10 |
| Sudoku Solving | TRM | **43.4%** | 14.2M | ~45ms | 10 (digits) | 9×9 |

### Training Performance
- **Maze**: Trained to target accuracy in ~30-60 minutes (GPU)
- **Optimization**: Adam with learning rate warmup
- **Deep Supervision**: Loss computed at all T recursive steps
- **Generalization**: Robust to unseen random mazes

## 🧪 Training & Evaluation

### Quick Training Example
```python
from src.train import train_model
from src.data.maze import MazeDataset

# Configure training
config = {
    'task': 'maze',
    'model_dim': 64,
    'num_heads': 4,
    'T': 2,  # Recursive depth
    'N_sup': 2,  # Deep supervision steps
    'batch_size': 128,
    'lr': 1e-3,
    'epochs': 100
}

# Train model
model = train_model(config)
```

### Evaluation
```bash
# Evaluate maze model
python check_maze_model.py --model outputs/maze_final.pt --num_samples 100

# Evaluate Sudoku model
python check_sudoku_model.py --model outputs/sudoku_final.pt
```

### Training Notebooks
- **[train_maze_10x10_FINAL.ipynb](notebooks/train_maze_10x10_FINAL.ipynb)**: Production maze training
- **[TRM_Colab_A100.ipynb](notebooks/TRM_Colab_A100.ipynb)**: Sudoku training on A100 GPU
- **[train_colab_maze_directions.ipynb](notebooks/train_colab_maze_directions.ipynb)**: Direction-based approach

## 🎨 Visualization Features

### Maze Grid Display
- **Cell Size**: 40×40px individual colored cells
- **Wall Color**: Dark gray (#334155)
- **Path Colors**: 
  - 🟢 Green (#10b981) = BFS optimal path
  - 🟣 Purple (#8b5cf6) = TRM predicted path
- **Markers**: 
  - ⭐ Blue = Start position
  - 🎯 Red = Goal position
  - 🔵 = Current TRM position
- **Animation**: 150ms step delays for smooth visualization

### Sudoku Grid
- **3×3 Box Borders**: Clear visual separation
- **Color Coding**:
  - Black (bold): Original clues
  - Purple: Algorithm-solved cells
  - Light gray: Empty cells
- **Animations**: Smooth fade-in effects (200ms)
- **Progress Tracking**: Real-time cells filled counter

## 🔧 Technical Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch 2.0+, transformers |
| **Frontend** | Streamlit 1.28+, HTML/CSS Grid |
| **Visualization** | Plotly 5.18+, matplotlib |
| **Data Processing** | NumPy, NetworkX, SymPy |
| **Algorithms** | BFS, Backtracking, Recursive Division, Cellular Automata |
| **Development** | Jupyter, W&B (Weights & Biases) |

## 📖 Documentation Hub

### Core Documentation
- **[PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)**: Comprehensive project overview and achievements
- **[architecture.md](docs/architecture.md)**: Deep dive into system and model architecture
- **[TRM_Concept_Explanation.txt](TRM_Concept_Explanation.txt)**: Theoretical foundation and mathematical formulation

### Demo & Presentation
- **[DEMO_GUIDE.md](DEMO_GUIDE.md)**: Step-by-step presentation instructions (15-20 min demo)
- **[presentation_outline.md](docs/presentation_outline.md)**: Structured presentation guide
- **[ui_design.md](docs/ui_design.md)**: UI specifications and design decisions

### Development
- **[test_cases.md](docs/test_cases.md)**: Comprehensive testing documentation
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)**: Deployment instructions
- **[product_backlog.md](docs/product_backlog.md)**: Feature tracking and sprint planning
- **[daily_scrum.md](docs/daily_scrum.md)**: Development progress log

## 🎯 Use Cases

### Educational
- 📚 Demonstrate recursive reasoning in AI
- 🎓 Teaching transformer architectures
- 🧠 Explaining "System 2 thinking" in ML
- 🔬 Research presentations

### Research
- 🏗️ Parameter-efficient model architectures
- 🔄 Recursive computation in transformers
- 📊 Deep supervision training techniques
- 🎯 Algorithmic task learning

### Development
- 🧪 Testing ground for TRM improvements
- 🎨 UI/UX design for ML demos
- 📈 Model performance visualization
- 🔍 Debugging and error analysis

## 🚀 Future Enhancements

### Model Improvements
- [ ] Larger maze sizes (20×20, 30×30)
- [ ] Additional puzzle types (8-puzzle, chess problems)
- [ ] Dynamic recursive depth (adaptive T)
- [ ] Ensemble models for improved accuracy
- [ ] Transfer learning experiments

### Demo Features
- [ ] User-uploadable mazes/puzzles
- [ ] Export animations as videos
- [ ] Multi-language support
- [ ] Dark/light theme toggle
- [ ] Model performance graphs
- [ ] Real-time fine-tuning interface

### Research Directions
- [ ] 3D maze navigation
- [ ] Longer sequence reasoning tasks
- [ ] Real-time online learning
- [ ] Attention visualization
- [ ] Scaling studies (parameters vs accuracy)

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

1. **Model Training**
   - Experiment with hyperparameters
   - Try different architectures
   - Train on new tasks

2. **Demo Enhancement**
   - UI/UX improvements
   - New visualization types
   - Performance optimizations

3. **Documentation**
   - Code examples
   - Tutorial notebooks
   - Architecture explanations

4. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks

## 📝 Known Limitations

1. **Maze Accuracy**: 64.5% means ~35% of mazes may not reach goal
   - **Workaround**: Generate new maze if TRM fails

2. **Grid Size Constraint**: Models trained only on specific sizes (10×10 maze, 9×9 Sudoku)
   - **Future**: Retrain on variable sizes

3. **Computational Cost**: Training requires GPU for reasonable speed
   - **Note**: Inference runs efficiently on CPU (~50ms)

4. **Browser Compatibility**: Demo optimized for Chrome/Edge
   - **Works**: Firefox, Safari (minor visual differences)

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Research**: Based on the Tiny Recursive Model paper ([2510.04871v1.pdf](2510.04871v1.pdf))
- **Inspiration**: Original demo.html design and visualization concepts
- **Framework**: Built with PyTorch and Streamlit
- **Community**: Open-source ML/AI community

## 📞 Project Information

- **Version**: 1.0 (Production Ready)
- **Status**: ✅ Complete and Stable
- **Last Updated**: January 6, 2026
- **Maintenance**: Stable state, minimal updates needed

---

<p align="center">
  <b>Built with ❤️ using PyTorch and Streamlit</b><br>
  <i>"Intelligence is not just knowing; it is the persistence of thinking."</i>
</p>
