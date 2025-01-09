import torch
import torch.nn as nn
import torchvision.models as models

class BaseCNN(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, freeze_layers=False):
        super(BaseCNN, self).__init__()
        
        if model_name == 'resnet18':
            # Lade ein vortrainiertes ResNet18 (ImageNet) aus TorchVision
            if pretrained:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet18(weights=None)
            
            # Entferne den letzten Fully-Connected-Layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
            # Option: Schichten einfrieren
            if freeze_layers:
                for param in self.model.parameters():
                    param.requires_grad = False

            # Die resultierende Ausgabe sollte nun die Form (B, 512, 1, 1) haben
            # flatten im forward()
            self.output_dim = 512
        
        else:
            raise ValueError(f"Model {model_name} not supported yet.")

    def forward(self, x):
        """
        x: Tensor der Form (B, 3, H, W)
        Returns: (B, 512) Feature Embeddings
        """
        features = self.model(x)  # -> (B, 512, 1, 1) bei ResNet18
        features = features.squeeze(-1).squeeze(-1)  # -> (B, 512)
        return features

if __name__ == "__main__":
    # Kleiner Test mit Optionen für Fine-Tuning
    dummy_input = torch.randn(2, 3, 96, 96)  # Anstatt 224x224 (typische ResNet-Input), hier 96x96

    # Option 1: Vollständiges Training (keine Schichten eingefroren)
    model_finetune = BaseCNN(model_name='resnet18', pretrained=True, freeze_layers=False)
    print("Vollständiges Training:")
    print(model_finetune)

    # Option 2: Eingefrorene Schichten (nur der Kopf wird trainiert)
    model_frozen = BaseCNN(model_name='resnet18', pretrained=True, freeze_layers=True)
    print("Eingefrorene Schichten:")
    print(model_frozen)

    # Test der Forward-Pass-Ausgabe
    out = model_finetune(dummy_input)
    print("Output shape (Vollständig trainiert):", out.shape)  # (2, 512)

    # Beispiel für optimierte Lernraten
    optimizer = torch.optim.Adam([
        {'params': model_finetune.model[:6].parameters(), 'lr': 1e-5},  # Niedrigere LR für frühere Schichten
        {'params': model_finetune.model[6:].parameters(), 'lr': 1e-4}   # Höhere LR für spätere Schichten
    ])
