import torch
import torch.nn as nn
import torchvision.models as models

class BaseCNN(nn.Module):
    """
    Erzeugt ein ResNet-Backbone für Feature-Extraktion, 
    optional mit (teilweisem) Freezing bestimmter Blöcke.

    Bei ResNet18 -> Feature-Dimension = 512 
    Bei ResNet50 -> Feature-Dimension = 2048
    """

    def __init__(
        self,
        model_name='resnet18',
        pretrained=True,
        freeze_blocks=None
    ):
        """
        Args:
            model_name: 'resnet18' oder 'resnet50'
            pretrained: bool, ob ImageNet-Gewichte
            freeze_blocks: None oder Liste von Block-Indizes, die eingefroren werden sollen
               - Beispiel: [0,1] => friere layer1 + layer2
               - Oder [0,1,2] => friere layer1 + layer2 + layer3
               - None => nix einfrieren
        """
        super(BaseCNN, self).__init__()

        # 1) Lade das gewünschte ResNet-Modell
        if model_name == 'resnet18':
            if pretrained:
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.resnet = models.resnet18(weights=None)
            self.output_dim = 512
        elif model_name == 'resnet50':
            if pretrained:
                self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            else:
                self.resnet = models.resnet50(weights=None)
            self.output_dim = 2048
        else:
            raise ValueError(f"Model {model_name} not supported. Choose 'resnet18' or 'resnet50'.")

        # 2) Entferne den letzten Fully-Connected-Layer
        #    => Alles behalten bis kurz vor self.resnet.fc
        #    => in children() sind: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        #    => [:-1] => wir behalten avgpool, entfernen fc
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # 3) Optional: Blöcke einfrieren
        self.freeze_blocks = freeze_blocks
        if freeze_blocks is not None:
            self._apply_block_freezing()

    def _apply_block_freezing(self):
        """
        Intern: friert die angegebenen ResNet-Blöcke in freeze_blocks ein.

        In self.resnet => (index => module):
          [0] = conv1 + bn1 + relu + maxpool
          [1] = layer1
          [2] = layer2
          [3] = layer3
          [4] = layer4
          [5] = avgpool
        => freeze_blocks=[0,1] => layer1+layer2 => bedeuten:
           block_idx +1 => actual_idx in self.resnet
           also block0=layer1 => actual_idx=1
           block1=layer2 => actual_idx=2
           block2=layer3 => actual_idx=3
           block3=layer4 => actual_idx=4
        """
        for block_idx in self.freeze_blocks:
            actual_idx = block_idx + 1  # da [1] = layer1
            if actual_idx < len(self.resnet):
                for param in self.resnet[actual_idx].parameters():
                    param.requires_grad = False
            else:
                raise ValueError(
                    f"Block index {block_idx} out of range for freeze_blocks. Must be in 0..3"
                )

    def forward(self, x):
        """
        x: Tensor (B, 3, H, W)
        Returns: (B, self.output_dim) Feature Embeddings
        """
        # self.resnet: conv1..layer4 + avgpool => => (B, out_dim,1,1)
        features = self.resnet(x)              # => (B, out_dim, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # => (B, out_dim)
        return features
