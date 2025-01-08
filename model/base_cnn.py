import torch
import torch.nn as nn
import torchvision.models as models

class BaseCNN(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        super(BaseCNN, self).__init__()
        
        if model_name == 'resnet18':
            # Lade ein vortrainiertes ResNet18 (ImageNet) aus TorchVision
            # ACHTUNG: Vortrainiert auf 224x224 RGB
            # -> Du kannst pretrained=False setzen, um nicht auf ImageNet-Weights zurückzugreifen
            if pretrained:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet18(weights=None)
            
            # Entferne den letzten FC-Layer
            # Original: self.model.fc = nn.Linear(512, 1000) für ResNet18
            # Wir wollen nur bis zum global average pooling
            # In PyTorch: list(self.model.children()) = [Conv1, BN1, ReLU, MaxPool, ..., layer4, avgpool, fc]
            # Wir nehmen alles außer fc:
            self.model = nn.Sequential(*list(self.model.children())[:-1])

            # Die resultierende Ausgabe sollte nun die Form (B, 512, 1, 1) haben
            # Wir flatten das später im forward()
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
    # Kleiner Test
    dummy_input = torch.randn(2, 3, 96, 96)  # Anstatt 224x224 (typische ResNet-Input), hier 96x96
    model = BaseCNN(model_name='resnet18', pretrained=False)
    out = model(dummy_input)
    print("Output shape:", out.shape)  # (2, 512)
