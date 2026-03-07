# 🧠 TRM Sudoku 7M Training - Colab Guide

## 📋 Overview

This notebook trains the **Tiny Recursive Model (TRM)** on **7 million Sudoku puzzles** to achieve the paper's benchmark of **87% test accuracy** using only **~5M parameters**.

**Paper**: _Less is More: Recursive Reasoning with Tiny Networks_ (arXiv:2510.04871)

---

## 🚀 Quick Start

### 1. Upload to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload Notebook**
3. Upload `TRM_Sudoku_7M_Production.ipynb`
4. Select **Runtime → Change runtime type → A100 GPU** (or H100 if available)

### 2. Run All Cells

Simply click **Runtime → Run all** or run cells sequentially.

**Estimated Training Time**: 12-24 hours on A100 GPU

---

## 📊 What the Notebook Does

### Section 1: Environment Setup
- Detects GPU (requires A100/H100 with 40GB+ VRAM)
- Installs dependencies: `datasets`, `wandb`, `torch`, `tqdm`
- Configures CUDA and mixed precision training

### Section 2: Configuration
- Sets all hyperparameters from the paper:
  - `N_sup=16` (full deep supervision)
  - `batch_size=768` (via gradient accumulation)
  - `lr=1e-4`, `weight_decay=1.0`
  - `60,000 iterations`
  - `EMA β=0.999`

### Section 3: Dataset Loading
- Streams **7M Sudoku puzzles** from HuggingFace (`sadimanna/1m-sudoku`)
- Memory-efficient loading (doesn't load all 7M into RAM)
- Automatic data validation and preprocessing

### Section 4: Model Architecture
- **TRM Model**: 2-layer transformer with 512-dim embeddings
- **Recursive mechanism**: T=3 cycles, n=6 latent updates
- **Parameter count**: ~5M (matches paper)
- Components:
  - RMSNorm (paper spec)
  - SwiGLU activation
  - Multi-head self-attention
  - Shared tiny network

### Section 5: Training Loop
- **Deep supervision** with N_sup=16 iterations
- **Gradient accumulation** (48 × 16 = 768 effective batch size)
- **Learning rate warmup** (2000 steps)
- **EMA** for stable generalization
- **Adaptive computation time (ACT)** for early halting
- **WandB logging** (optional)
- **Checkpointing** every 5000 steps

### Section 6: Evaluation
- Loads best checkpoint with EMA weights
- Evaluates on 10K test puzzles
- Reports:
  - **Cell accuracy** (per-cell correctness)
  - **Grid accuracy** (all 81 cells correct) - **Target: 87%**

### Section 7: Download
- Downloads trained model checkpoint
- Available for deployment

---

## ⚙️ Configuration Options

You can modify these in **Section 2**:

```python
CONFIG = {
    # Quick test run (5 minutes)
    'max_iters': 500,
    'max_examples': 1000,
    
    # OR Full training (12-24 hours)
    'max_iters': 60000,
    'max_examples': 7000000,
    
    # Memory optimization
    'batch_size': 24,  # Lower if OOM errors
    'gradient_accumulation': 32,  # Increase proportionally
    
    # WandB tracking
    'use_wandb': True,  # Set False to disable
}
```

---

## 📈 Expected Results

Based on paper benchmarks:

### Training Progress
| Iteration | Loss | Cell Acc | Grid Acc |
|-----------|------|----------|----------|
| 5,000 | ~1.5 | ~60% | ~5% |
| 20,000 | ~0.8 | ~80% | ~35% |
| 40,000 | ~0.4 | ~90% | ~65% |
| **60,000** | **~0.2** | **~95%** | **~87%** |

### Final Metrics (Paper Target)
- **Grid Accuracy**: 87.4%
- **Cell Accuracy**: ~95%
- **Parameters**: 5M
- **Training Time**: 12-24 hours (A100)

---

## 🔧 Troubleshooting

### Out of Memory (OOM) Errors

**Symptom**: CUDA out of memory during training

**Solution 1** - Reduce batch size:
```python
CONFIG['batch_size'] = 24  # Down from 48
CONFIG['gradient_accumulation'] = 32  # Up from 16
```

**Solution 2** - Reduce model size (will affect accuracy):
```python
CONFIG['dim'] = 384  # Down from 512
```

**Solution 3** - Use H100 (80GB VRAM) instead of A100 (40GB)

### Slow Training

**Symptom**: Less than 1 iteration/second

**Checks**:
1. Verify A100/H100 is selected: `Runtime → Change runtime type`
2. Check GPU utilization: Run `!nvidia-smi` in a cell
3. Verify mixed precision is enabled: `device == 'cuda'`

**Expected Speed**:
- A100: ~2 seconds/iteration → 33 hours total
- H100: ~1 second/iteration → 16 hours total

### Loss Not Decreasing

**Symptom**: Loss stays constant or increases

**Checks**:
1. Verify learning rate warmup is active (first 2000 steps)
2. Check gradient clipping: Should be 1.0
3. Verify EMA is updating: Look for "✓ Optimizer and EMA initialized"

**Restart Training** if divergence occurs:
- Loss > 10.0 after 10K iterations = divergence
- Start fresh or reduce learning rate to 5e-5

### Dataset Loading Errors

**Symptom**: "Dataset not found" or connection errors

**Solution**:
```python
# Try alternative dataset
CONFIG['dataset_name'] = 'Ritvik19/Sudoku-Dataset'

# Or download manually
!wget https://huggingface.co/datasets/sadimanna/1m-sudoku/resolve/main/data.zip
!unzip data.zip
```

---

## 📊 Monitoring Training

### WandB Dashboard (Recommended)

If `use_wandb=True`:

1. You'll be prompted to login (first time only)
2. Visit the printed WandB URL
3. Monitor in real-time:
   - Training loss curve
   - Accuracy progression
   - Supervision steps (ACT early halting)
   - GPU utilization

### Console Output

Without WandB, watch the progress bar:

```
Training: 45%|████▌     | 27000/60000 [15:00:00<18:30:00, 2.00s/it, loss=0.4123, acc=0.845, sup=8.2]
```

- `loss`: Lower is better (target: ~0.2)
- `acc`: Cell accuracy (target: ~95%)
- `sup`: Average supervision steps (should be 8-12)

---

## 💾 Checkpoints

Checkpoints are saved automatically:

- **Every 5000 iterations**: `/content/trm_checkpoint_{step}.pt`
- **Final model**: `/content/trm_final.pt`

### Resume from Checkpoint

If training is interrupted, modify Section 5:

```python
# Load checkpoint
checkpoint = torch.load('/content/trm_checkpoint_25000.pt')
model.load_state_dict(checkpoint['model'])
embedding.load_state_dict(checkpoint['embedding'])
optimizer.load_state_dict(checkpoint['optimizer'])
ema_params = checkpoint['ema_params']
start_step = checkpoint['step']

# Continue training from start_step
for step in range(start_step, CONFIG['max_iters']):
    # ... rest of training loop
```

---

## 🎯 Using the Trained Model

After training completes:

```python
# Load checkpoint
checkpoint = torch.load('/content/trm_final.pt')
model.load_state_dict(checkpoint['model'])
embedding.load_state_dict(checkpoint['embedding'])

# Apply EMA weights
for name, p in model.named_parameters():
    key = f'model_{name}'
    if key in checkpoint['ema_params']:
        p.data = checkpoint['ema_params'][key]

model.eval()

# Solve a Sudoku puzzle
puzzle = torch.tensor([...])  # Your 81-digit puzzle
x = embedding(puzzle.unsqueeze(0))
y = embedding(puzzle.unsqueeze(0))
z = torch.zeros_like(x)

# Run N_sup iterations
for _ in range(16):
    (y, z), y_hat, q_hat = model(x, y, z)
    if torch.sigmoid(q_hat) > 0.95:
        break

solution = y_hat.argmax(-1)
print(solution.reshape(9, 9))
```

---

## 📚 Paper Implementation Details

This notebook implements **Algorithm 3** from the paper:

### Key Differences from HRM
1. **Single network** instead of two (fL and fH)
2. **T-1 no-grad warmups** + 1 gradient cycle (no fixed-point assumption)
3. **Full deep supervision** (all N_sup=16 steps get gradients)
4. **MLP-free** for Sudoku (uses self-attention)
5. **Strong EMA** (β=0.999 vs typical 0.995)

### Hyperparameter Justification

From paper Appendix:

- **batch_size=768**: Large batch stabilizes deep supervision
- **weight_decay=1.0**: Strong L2 prevents overfitting on small data
- **N_sup=16**: Full supervision vs HRM's ACT-reduced ~2 steps
- **warmup=2K**: Stabilizes initial recursion gradients
- **EMA=0.999**: Smooths supervision step oscillations

---

## 🔬 Next Steps After Training

1. **Evaluate on Sudoku-Extreme**:
   - Download benchmark from paper repository
   - Run evaluation script
   - Compare to 87.4% paper result

2. **Transfer to ARC-AGI**:
   - Modify for 30×30 grids
   - 10-color vocabulary
   - Retrain on ARC tasks

3. **Model Analysis**:
   - Visualize attention patterns
   - Extract latent z-traces
   - Study recursive refinement

4. **Deployment**:
   - Export to ONNX for production
   - Quantize to INT8 for edge devices
   - Build API endpoint

---

## 📞 Support

**Issues?**
- Check Troubleshooting section above
- Review paper Appendix for hyperparameter details
- Verify all paper specifications are matched

**Questions about the paper?**
- arXiv: [2510.04871](https://arxiv.org/abs/2510.04871)
- Original HRM: arXiv:2506.21734

---

## ✅ Checklist Before Running

- [ ] **GPU**: A100 (40GB) or H100 (80GB) selected
- [ ] **Runtime**: High-RAM option selected in Colab
- [ ] **Time**: Have 12-24 hours GPU availability
- [ ] **Storage**: ~10GB free for checkpoints
- [ ] **WandB** (optional): Account created for tracking
- [ ] **Dataset**: Verified HuggingFace connection works

**Ready to train?** Run all cells! 🚀

---

**Good luck achieving 87% accuracy!**
