import torch
import torch.nn as nn
import torchvision.models as models

class BaseCNN(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        super(BaseCNN, self).__init__()
        
        # Lade ein vortrainiertes Model
        # Variationen: resnet18, resnet50, densenet121 usw.
        if model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Entfernen der letzten voll verbundene Schicht
            # Original: self.model.fc = nn.Linear(512, 1000) bei ResNet-18
            # FC muss entfernt werden und statt dessen nur Features bis zum global average pooling behalten
            self.model = nn.Sequential(*list(self.model.children())[:-1])  
            # Die resultierende Ausgabe sollte nun [Batch, 512, 1, 1] sein
            # Durch Squeeze -> [Batch, 512]
            
            self.output_dim = 512  # Dimension des Feature-Vektors
            
        else:
            raise ValueError(f"Model {model_name} not supported yet.")

    def forward(self, x):
        """
        x: Tensor mit Form (B, C=3, H, W)
        """
        features = self.model(x)  # (B, 512, 1, 1) bei ResNet-18
        features = features.squeeze()  # (B, 512)
        return features

if __name__ == "__main__":
    # Test mit Dummy-Daten:
    # Erstellen eines Dummy-Patchs der Größe (Batch=2, Channels=3, Height=224, Width=224)
    # Height und Width kann später an Patch-Größen anpasst werden
    dummy_input = torch.randn(2, 3, 224, 224)
    model = BaseCNN(model_name='resnet18', pretrained=False)
    out = model(dummy_input)
    print("Output shape:", out.shape)  # sollte (2, 512) sein
