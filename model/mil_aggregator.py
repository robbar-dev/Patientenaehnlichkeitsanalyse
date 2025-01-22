import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class AttentionMILAggregator(nn.Module):
    """
    Gated-Attention-MIL-Implementierung (nach Ilse et al. 2018) mit Dropout.
    Für jeden Patch berechnen wir einen Score und gewichten die Patch-Features
    mittels Softmax. Dadurch lernt das Modell, welche Patches wichtiger sind.

    Kerngedanke:
      - 'Gated' bedeutet, wir haben zwei Pfade (Tanh und Sigmoid),
        die elementweise multipliziert werden (Gate).
      - Anschließend berechnen wir daraus einen linearen Score pro Patch,
        normalisieren via Softmax und summieren gewichtet.
    """

    def __init__(self, in_dim=2048, hidden_dim=128, dropout=0.2):
        """
        Args:
          in_dim: Dimension der Patch-Features (z.B. 512 für ResNet18)
          hidden_dim: Größe der Zwischenschicht in der Attention-MLP
          dropout: Dropout-Rate für das Gating-Netzwerk
        """
        super().__init__()
        logging.info(f"Using Aggregator: {self.__class__.__name__}")

        # U und V sind die zwei Pfade (Tanh, Sigmoid)
        #    U: Linear(in_dim -> hidden_dim)
        #    V: Linear(in_dim -> hidden_dim)
        # plus Dropout-Layer nach dem gating
        self.u_layer = nn.Linear(in_dim, hidden_dim)   # Pfad U -> Tanh
        self.v_layer = nn.Linear(in_dim, hidden_dim)   # Pfad V -> Sigmoid

        # Dropoutlayer (wird auf das Zwischenergebnis h angewendet)
        self.dropout = nn.Dropout(p=dropout)

        # Letzter Layer W: map (hidden_dim) -> (1) => Score
        self.w_layer = nn.Linear(hidden_dim, 1)

    def forward(self, patch_embs):
        """
        Args:
          patch_embs: Tensor der Form (N, in_dim),
                      N = #Patches pro Patient, in_dim = Feature-Dimension

        Returns:
          patient_emb: (1, in_dim)-Tensor, das das aggregated embedding enthält.
        """
        # patch_embs shape: (N, in_dim)

        # 1) Tanh-Pfad
        u = torch.tanh(self.u_layer(patch_embs))  # => (N, hidden_dim)

        # 2) Sigmoid-Pfad
        v = torch.sigmoid(self.v_layer(patch_embs))  # => (N, hidden_dim)

        # 3) Gating => elementweise Multiplikation
        # h shape: (N, hidden_dim)
        h = u * v

        # 4) Dropout
        #    Regulierung gegen Overfitting
        h = self.dropout(h)  # => (N, hidden_dim)

        # 5) End-Scoring => w_layer(h) => shape (N,1), squeeze -> (N)
        scores = self.w_layer(h).squeeze(-1)  # => (N)

        # 6) Softmax => normalisierte Gewichte alpha
        alpha = F.softmax(scores, dim=0)      # => (N)

        # 7) Gewichtete Summe über Patch-Embeddings
        #    patch_embs * alpha => (N, in_dim), summiere -> (in_dim)
        weighted_emb = patch_embs * alpha.unsqueeze(-1)  # => (N, in_dim)
        patient_emb  = torch.sum(weighted_emb, dim=0)     # => (in_dim,)

        # 8) (1, in_dim) zurückgeben
        return patient_emb.unsqueeze(0)
