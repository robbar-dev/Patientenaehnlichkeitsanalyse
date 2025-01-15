import torch
import torch.nn as nn

class MeanPoolingAggregator(nn.Module):
    """
    Ein einfacher Aggregator, der Patch-Embeddings via mean pooling aggregiert.
    patch_embs shape: (N, d)
    Rückgabe: (1, d)
    """
    def __init__(self):
        super().__init__()
        # Kein Parameter nötig, rein "parametrischer" Aggregator

    def forward(self, patch_embs):
        """
        Args:
          patch_embs: Tensor der Form (N, d)
        Returns:
          patient_emb: shape (1, d)
        """
        # patch_embs => (N, d)
        # mean over dim=0 => (d,)
        mean_emb = torch.mean(patch_embs, dim=0)  # => (d,)
        # unsqueeze(0) => (1, d)
        return mean_emb.unsqueeze(0)
