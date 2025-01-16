import torch
import torch.nn as nn
import logging




class MaxPoolingAggregator(nn.Module):
    """
    Ein einfacher Aggregator, der Patch-Embeddings via max pooling aggregiert.
    patch_embs shape: (N, d)
    Rückgabe: (1, d)
    """
    def __init__(self):
        super().__init__()
        logging.info(f"Using Aggregator: {self.__class__.__name__}")
    

    def forward(self, patch_embs):
        """
        Args:
          patch_embs: Tensor der Form (N, d)
        Returns:
          patient_emb: shape (1, d)
        """
        # patch_embs => (N, d)
        # max over dim=0 => (d,)
        max_emb = torch.max(patch_embs, dim=0).values  # => (d,)
        # unsqueeze(0) => (1, d)
        return max_emb.unsqueeze(0)
