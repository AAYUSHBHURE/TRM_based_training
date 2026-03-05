"""
TRM Dataset Loader (Simplified)

Returns raw token tensors - embedding happens in model forward pass.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import Dict


class TRMDataset(Dataset):
    """TRM dataset for Sudoku training."""
    
    def __init__(
        self,
        data_path: str,
        domain: str = "sudoku",
        embed_dim: int = 512,
        max_seq_len: int = 128,
        vocab_size: int = 10,
        **kwargs
    ):
        self.data_path = Path(data_path)
        self.domain = domain
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Sudoku processing
        puzzle = torch.tensor(item["puzzle"], dtype=torch.long)  # [81]
        solution = torch.tensor(item["solution"], dtype=torch.long)  # [81]
        
        # Pad to max_seq_len
        x_tokens = torch.zeros(self.max_seq_len, dtype=torch.long)
        x_tokens[:81] = puzzle
        
        # Random initial guess (1-9 for blanks)
        y_init_tokens = puzzle.clone()
        blank_mask = puzzle == 0
        y_init_tokens[blank_mask] = torch.randint(1, 10, (blank_mask.sum(),))
        y_init = torch.zeros(self.max_seq_len, dtype=torch.long)
        y_init[:81] = y_init_tokens
        
        # Solution
        y_true = torch.zeros(self.max_seq_len, dtype=torch.long)
        y_true[:81] = solution
        
        return {
            "x_tokens": x_tokens,
            "y_init_tokens": y_init,
            "y_true": y_true,
            "seq_len": torch.tensor(81),
        }


def create_dataloader(
    data_path: str,
    domain: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,  # 0 for Windows compatibility
    **dataset_kwargs
) -> DataLoader:
    """Create TRM DataLoader."""
    dataset = TRMDataset(data_path, domain, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
