import os
import logging
import torch
import torch.nn as nn

class BinaryTripletTester(nn.Module):
    """
    Test-Trainer für binäre Klassifikation
    Enthält:
      - CNN-Backbone
      - Aggregator
      - Klassifikationskopf
    Lädt Checkpoint und kann compute_patient_embedding(...) aufrufen.
    KEINE Trainingsmethoden, KEINE Sampler, KEIN BCE/Triplet-Loss.
    """

    def __init__(
        self,
        data_root,
        device='cuda',
        model_name='resnet18',
        freeze_blocks=None,
        aggregator_name='mil',
        agg_hidden_dim=128,
        agg_dropout=0.4,
        roi_size=(96,96,3),
        overlap=(10,10,1)
    ):
        """
        Args:
          data_root: Basis-Verzeichnis mit den Bilddaten.
          device: 'cuda' oder 'cpu'
          model_name: z.B. 'resnet18' (BaseCNN)
          freeze_blocks: z.B. [0,1] oder None
          aggregator_name: 'mil', 'max', 'mean'
          agg_hidden_dim: nur relevant bei aggregator='mil'
          agg_dropout: nur relevant bei aggregator='mil'
          roi_size: Patch-Größe
          overlap: Overlap der Patches
        """
        super().__init__()
        self.data_root = data_root
        self.device = device
        self.model_name = model_name
        self.freeze_blocks = freeze_blocks
        self.aggregator_name = aggregator_name
        self.agg_hidden_dim = agg_hidden_dim
        self.agg_dropout = agg_dropout
        self.roi_size = roi_size
        self.overlap = overlap

        from model.base_cnn import BaseCNN
        self.base_cnn = BaseCNN(
            model_name=self.model_name,
            pretrained=True,
            freeze_blocks=self.freeze_blocks
        ).to(self.device)

        from model.mil_aggregator import AttentionMILAggregator
        from model.max_pooling import MaxPoolingAggregator
        from model.mean_pooling import MeanPoolingAggregator

        if aggregator_name == 'mil':
            self.mil_agg = AttentionMILAggregator(
                in_dim=512,
                hidden_dim=self.agg_hidden_dim,
                dropout=self.agg_dropout
            ).to(self.device)
        elif aggregator_name == 'max':
            self.mil_agg = MaxPoolingAggregator().to(self.device)
        elif aggregator_name == 'mean':
            self.mil_agg = MeanPoolingAggregator().to(self.device)
        else:
            raise ValueError(f"Unbekannte aggregator_name={aggregator_name}, nutze 'mil','max','mean'.")

        # Klassifikationskopf 
        self.classifier = nn.Linear(512, 1).to(self.device)

        logging.info("[BinaryTripletTester] Inference-Modell erstellt.")
        logging.info(f"Backbone={model_name}, aggregator={aggregator_name}, freeze_blocks={freeze_blocks}, "
                     f"agg_hidden_dim={agg_hidden_dim}, agg_dropout={agg_dropout}")

    def _forward_patient(self, pid, study_yr):
        from torch.utils.data import DataLoader
        from training.data_loader import SinglePatientDataset

        # KEIN Training 
        self.base_cnn.eval()
        self.mil_agg.eval()
        self.classifier.eval()

        ds = SinglePatientDataset(
            data_root=self.data_root,
            pid=pid,
            study_yr=study_yr,
            roi_size=self.roi_size,
            overlap=self.overlap,
            skip_factor=1,
            do_augmentation=False  # keine Augmentation im Test!!
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        patch_embs = []
        with torch.no_grad():
            for patch_t in loader:
                patch_t = patch_t.to(self.device)
                emb = self.base_cnn(patch_t)  # => (B,512)
                patch_embs.append(emb)

        if len(patch_embs) == 0:
            dummy = torch.zeros((1,512), device=self.device)
            logits= self.classifier(dummy)  # => (1,1)
            return dummy, logits

        patch_embs = torch.cat(patch_embs, dim=0)  # => (N,512)
        patient_emb = self.mil_agg(patch_embs)      # => (1,512)
        logits = self.classifier(patient_emb)       # => (1,1)
        return patient_emb, logits

    def compute_patient_embedding(self, pid, study_yr):
        """
        Liefert das (1,512)-Embedding für einen Patienten 
        """
        emb, logits = self._forward_patient(pid, study_yr)
        return emb  # => (1,512)

    def compute_patient_logits(self, pid, study_yr):
        """
        binären Klassifikations-Logit (1,1) 
        """
        emb, logits = self._forward_patient(pid, study_yr)
        return logits  # => (1,1)

    def load_checkpoint(self, path):
        """
        Lädt Gewichte aus dem gespeicherten Checkpoint 
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint {path} existiert nicht!")

        ckpt= torch.load(path, map_location=self.device)
        self.base_cnn.load_state_dict(ckpt['base_cnn'])
        self.mil_agg.load_state_dict(ckpt['mil_agg'])
        self.classifier.load_state_dict(ckpt['classifier'])
        logging.info(f"[BinaryTripletTester] Checkpoint geladen von {path}.")
