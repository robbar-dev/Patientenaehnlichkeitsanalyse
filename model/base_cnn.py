import torch
import torch.nn as nn
import torchvision.models as models

class BaseCNN(nn.Module):
    """
    Erzeugt ein ResNet18-Backbone für Feature-Extraktion, 
    optional mit (teilweisem) Freezing bestimmter Blöcke.

    Feature-Dimension = 512 (ResNet18-Ende, ohne FC)
    """

    def __init__(self, 
                 model_name='resnet18',
                 pretrained=True,
                 freeze_blocks=None):
        """
        Args:
            model_name: z.B. 'resnet18'
            pretrained: bool, ob ImageNet-Gewichte
            freeze_blocks: None oder Liste von Block-Indizes, die eingefroren werden sollen
               - Beispiel: [0,1] => friere die ersten 2 Blöcke von ResNet18
               - Oder [0,1,2] => friere die ersten 3 Blöcke
               - None => gar nichts einfrieren
        """
        super(BaseCNN, self).__init__()

        if model_name == 'resnet18':
            # 1) Lade ein vortrainiertes ResNet18
            if pretrained:
                self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.resnet = models.resnet18(weights=None)  # random init
            
            # 2) Entferne den letzten Fully-Connected-Layer
            #    => Alles behalten bis kurz vor self.resnet.fc
            #    => in children() sind: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
            #    => [:-1] => avgpool behalten, fc raus
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
            
            self.output_dim = 512  # ResNet-18 -> 512 Feature-Dimension

        else:
            raise ValueError(f"Model {model_name} not supported yet.")

        # 3) Optional: Friere bestimmte Blöcke
        #    ResNet-18-Blocks in self.resnet:
        #      - conv1 + bn1 + relu + maxpool = 'stem'
        #      - layer1 (Block0)
        #      - layer2 (Block1)
        #      - layer3 (Block2)
        #      - layer4 (Block3)
        #      - avgpool (global pool)
        #    Wir definieren indices: 0=layer1, 1=layer2, 2=layer3, 3=layer4
        #
        #    => freeze_blocks=[0,1] => freeze layer1 + layer2
        #    => freeze_blocks=None => nix einfrieren
        #
        #    Achtung: 'stem' (conv1,bn1,...) kann man extra behandeln.
        #    Hier ein simpler approach: 
        self.freeze_blocks = freeze_blocks
        if freeze_blocks is not None:
            self._apply_block_freezing()

    def _apply_block_freezing(self):
        """
        Intern: friert die angegebenen Blocks in freeze_blocks ein.
        """
        # In self.resnet => [0] = conv1, bn1, relu, maxpool
        #                 [1] = layer1
        #                 [2] = layer2
        #                 [3] = layer3
        #                 [4] = layer4
        #                 [5] = avgpool
        # => Indizes offset um 1 => d.h. Block0 = layer1 => index=1
        #
        # Real: 
        # [0] = conv1+bn1+relu+maxpool
        # [1] = layer1
        # [2] = layer2
        # [3] = layer3
        # [4] = layer4
        # [5] = avgpool
        # Definition von freeze_blocks=[0,1,2,3] als 4 resnet-blocks
        # => Im code 1=layer1,2=layer2,3=layer3,4=layer4
        # => also block_i +1 => actual index

        for block_idx in self.freeze_blocks:
            actual_idx = block_idx + 1  # da [1]=layer1
            if actual_idx < len(self.resnet):
                for param in self.resnet[actual_idx].parameters():
                    param.requires_grad = False
            else:
                raise ValueError(f"Block index {block_idx} out of range for ResNet18 (0..3)")

    def forward(self, x):
        """
        x: Tensor der Form (B, 3, H, W)
        Returns: (B, 512) Feature Embeddings
        """
        # self.resnet = conv1..layer4 + avgpool => => (B,512,1,1) am Ende
        features = self.resnet(x)           # => (B,512,1,1)
        features = features.squeeze(-1).squeeze(-1)  # => (B, 512)
        return features

if __name__ == "__main__":
    import torch

    # Kleiner Test
    dummy_input = torch.randn(2, 3, 96, 96)

    # Fall 1: pretrained, freeze_blocks=[0,1] => layer1 & layer2 werden eingefroren
    model_frozen = BaseCNN(
        model_name='resnet18',
        pretrained=True,
        freeze_blocks=[0,1]
    )
    print("Teil-gefrorenes Modell (layer1,layer2):")
    print(model_frozen)

    out = model_frozen(dummy_input)
    print("Output shape:", out.shape)  # (2, 512)

    # Fall 2: kein freeze
    model_unfrozen = BaseCNN(
        model_name='resnet18',
        pretrained=True,
        freeze_blocks=None
    )
    print("Komplett trainierbares Modell:")
    print(model_unfrozen)
