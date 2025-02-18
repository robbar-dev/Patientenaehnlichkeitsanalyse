import random
import torch

DEBUG = False

def parse_combo_string(combo_str):
    """
    Für binäre Klassifikation:
      combo_str '1-0-0'  => abnormale Lunge -> 1 
      normale Lunge => Klasse '0'
    """
    if combo_str.startswith('1-0-0'):
        return "1"
    else:
        return "0"

class HardNegativeBinaryTripletSampler:
    """
    - Offline-Berechnung der Embeddings für alle Patienten via 'trainer.compute_patient_embedding'
    - Pro Anchor:
        Anchor => (pid, study_yr, combo, multi_label)
        Positive => gleiche combo -> also 1 oder 0 
        Negative => "härtester" (embedding-dist kleinster) Patient aus einer ANDEREN combo
    """

    def __init__(
        self,
        df,
        trainer,
        num_triplets=1000,
        device='cuda'
    ):
        """
        Args:
          df: Pandas DataFrame mit Spalten [pid, study_yr, combination]
          trainer: TripletTrainer (der compute_patient_embedding bereitstellt)
          num_triplets: Gewünschte Anzahl Triplets pro Epoche
          device: 'cuda' oder 'cpu'
        """
        self.df = df.copy()
        self.trainer = trainer
        self.num_triplets = num_triplets
        self.device = device

        # Offline-Embeddings sammeln
        self.patient_info_list = []
        print("[HardNegSampler] Compute Embeddings for all patients ...")
        with torch.no_grad():
            for i, row in self.df.iterrows():
                pid = row['pid']
                sy  = row['study_yr']
                combo_str = row['combination']

                combo_bin = parse_combo_string(combo_str)
                if combo_bin == "1":
                    multi_label_vec = [1]
                else:
                    multi_label_vec = [0]

                emb = trainer.compute_patient_embedding(pid, sy)  # -> shape(1,d)
                emb = emb.squeeze(0).cpu()                        # -> shape(d,)

                self.patient_info_list.append({
                    'pid': pid,
                    'study_yr': sy,
                    'combo': combo_bin,          # "0" oder "1"
                    'multi_label': multi_label_vec,  # [0] oder [1]
                    'embedding': emb
                })

        # combo_dict => "0" -> Liste[patient_info], "1" -> Liste[patient_info]
        self.combo_dict = {}
        for pinfo in self.patient_info_list:
            c = pinfo['combo']  # "0" oder "1"
            if c not in self.combo_dict:
                self.combo_dict[c] = []
            self.combo_dict[c].append(pinfo)

        self.all_combos = list(self.combo_dict.keys())  # ["0","1"]

        # Tensor mit Embeddings => shape(N,d)
        self.emb_tensor = torch.stack(
            [p['embedding'] for p in self.patient_info_list], 
            dim=0
        )  # => (N,d)

        # pid->Index Mapping
        self.pid_to_index = {}
        for idx, pinfo in enumerate(self.patient_info_list):
            self.pid_to_index[pinfo['pid']] = idx

        # Distanzmatrix (N,N)
        self.dist_matrix = torch.cdist(
            self.emb_tensor.unsqueeze(0),
            self.emb_tensor.unsqueeze(0),
            p=2
        ).squeeze(0)  # => (N,N)

        self.used_triplets = set()

    def __iter__(self):
        """
        generiert num_triplets Triplets (Anchor, Positive, Hard Negative)
        """
        count = 0
        attempts = 0
        max_attempts = self.num_triplets * 20

        while count < self.num_triplets and attempts < max_attempts:
            attempts += 1

            anchor_combo = random.choice(self.all_combos)
            anchor_list = self.combo_dict[anchor_combo]
            if len(anchor_list) < 1:
                continue

            anchor_info = random.choice(anchor_list)
            pos_info = random.choice(anchor_list)

            a_idx = self.pid_to_index[anchor_info['pid']]
            dists_a = self.dist_matrix[a_idx]  # => shape(N,)

            # Sortiere nach Distanz
            sorted_indices = torch.argsort(dists_a, dim=0)

            neg_info = None
            for idx_neg in sorted_indices:
                idx_neg = idx_neg.item()
                if idx_neg == a_idx:
                    continue
                c_neg = self.patient_info_list[idx_neg]['combo']
                if c_neg != anchor_combo:
                    neg_info = self.patient_info_list[idx_neg]
                    break

            if neg_info is None:
                continue

            # Triplet-ID, um Duplikate zu verhindern
            trip_id = (
                anchor_info['pid'],
                pos_info['pid'],
                neg_info['pid'],
                anchor_combo,
                neg_info['combo']
            )
            if trip_id in self.used_triplets:
                continue

            self.used_triplets.add(trip_id)
            yield (anchor_info, pos_info, neg_info)
            count += 1

        if DEBUG:
            print(f"[HardNegativeSampler] Generated {count} triplets out of {attempts} attempts.")

    def __len__(self):
        return self.num_triplets

    def reset_epoch(self):
        self.used_triplets = set()
