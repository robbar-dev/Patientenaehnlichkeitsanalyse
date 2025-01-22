import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2):
        """
        Args:
        - margin (float): Der Abstand, um den die negativen Paare größer als die positiven Paare sein sollen.
        - p (int): Normtyp für die Distanzberechnung (Standard: 2 für die euklidische Norm).
        """
        super(TripletLoss, self).__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=p)
    
    def forward(self, anchor, positive, negative, normalize_embeddings=False):
        """
        Args:
        - anchor (torch.Tensor): Tensor der Form (Batch, Embedding_dim), der die Anker-Embeddings enthält.
        - positive (torch.Tensor): Tensor der Form (Batch, Embedding_dim), der die positiven Embeddings enthält.
        - negative (torch.Tensor): Tensor der Form (Batch, Embedding_dim), der die negativen Embeddings enthält.
        - normalize_embeddings (bool): Gibt an, ob die Embeddings vor der Berechnung normiert werden sollen.
        
        Returns:
        - loss (torch.Tensor): Skalare Loss-Wert.
        """
        if normalize_embeddings:
            anchor = nn.functional.normalize(anchor, p=2, dim=1)
            positive = nn.functional.normalize(positive, p=2, dim=1)
            negative = nn.functional.normalize(negative, p=2, dim=1)
        
        # Berechnung der Triplet Loss
        loss = self.loss_fn(anchor, positive, negative)
        
        # Debugging-Informationen (optional)
        with torch.no_grad():
            distance_positive = torch.norm(anchor - positive, p=2, dim=1)
            distance_negative = torch.norm(anchor - negative, p=2, dim=1)
            print(f"Positive distances: {distance_positive}")
            print(f"Negative distances: {distance_negative}")
        
        return loss

if __name__ == "__main__":
    # Test mit Dummy-Embeddings
    batch_size = 4
    embedding_dim = 2048
    
    # Erstellen von Dummy-Daten
    anchor = torch.randn(batch_size, embedding_dim)
    positive = torch.randn(batch_size, embedding_dim)
    negative = torch.randn(batch_size, embedding_dim)
    
    # Instanziierung der TripletLoss-Klasse
    loss_fn = TripletLoss(margin=1.0)
    
    # Test ohne Normierung
    print("### Ohne Normierung ###")
    loss = loss_fn(anchor, positive, negative, normalize_embeddings=False)
    print("Triplet Loss (ohne Normierung):", loss.item())
    
    # Test mit Normierung
    print("\n### Mit Normierung ###")
    loss = loss_fn(anchor, positive, negative, normalize_embeddings=True)
    print("Triplet Loss (mit Normierung):", loss.item())

