import random
import torch

DEBUG = False

def parse_combo_string(combo_str):
    """
    '0-1-1' -> (0,1,1)
    """
    return tuple(int(x) for x in combo_str.split('-'))

class HardNegativeTripletSampler:
    """
    - Offline-Berechnung der Embeddings für alle Patienten via 'trainer.compute_patient_embedding'
    - Pro Anchor wählt man:
        Anchor => (pid, study_yr, combination, multi_label)
        Positive => gleicher combo
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
          trainer: dein TripletTrainer (muss compute_patient_embedding(...) haben)
          num_triplets: wie viele Triplets wir pro Epoche ausgeben
          device: 'cuda' oder 'cpu'
        """
        self.df = df.copy()
        self.trainer = trainer
        self.num_triplets = num_triplets
        self.device = device

        # 1) Liste aller Patienten => embedding
        #    Speichere in patient_info: { 'pid','study_yr','combo','multi_label','embedding'}
        self.patient_info_list = []
        print("[HardNegSampler] Compute Embeddings for all patients ...")
        with torch.no_grad():
            for i, row in self.df.iterrows():
                pid = row['pid']
                sy  = row['study_yr']
                combo = row['combination']

                # Parse combo => multi_label
                combo_tuple = parse_combo_string(combo)  # z.B. (0,1,1)
                multi_label_vec = list(combo_tuple)       # z.B. [0,1,1]

                emb = trainer.compute_patient_embedding(pid, sy)
                # emb shape (1, d)
                emb = emb.squeeze(0).cpu()

                self.patient_info_list.append({
                    'pid': pid,
                    'study_yr': sy,
                    'combo': combo,
                    'multi_label': multi_label_vec,  # <--- WICHTIG: Jetzt vorhanden
                    'embedding': emb
                })

        # 2) In combo_dict => combo_str -> Liste von patient_info
        self.combo_dict = {}
        for pinfo in self.patient_info_list:
            c = pinfo['combo']
            if c not in self.combo_dict:
                self.combo_dict[c] = []
            self.combo_dict[c].append(pinfo)

        self.all_combos = list(self.combo_dict.keys())

        # 3) Wir erstellen eine Liste (N,d) => Distanzberechnung
        #    patient_embeddings: shape (N, d)
        self.emb_tensor = torch.stack([p['embedding'] for p in self.patient_info_list], dim=0)
        # => (N,d)

        # dictionary: (pid->Index) und Index->(pid, synergy, combo, multi_label, embedding)
        self.pid_to_index = {}
        for idx, pinfo in enumerate(self.patient_info_list):
            self.pid_to_index[pinfo['pid']] = idx

        # 4) Distanzmatrix berechnen => (N,N)
        self.dist_matrix = torch.cdist(
            self.emb_tensor.unsqueeze(0), 
            self.emb_tensor.unsqueeze(0), 
            p=2
        ).squeeze(0)
        # => shape (N,N)

        # 5) used_triplets => pro Epoche unique triplets
        self.used_triplets = set()

    def __iter__(self):
        """
        generiert num_triplets Triplets
        Hard Negative => wir suchen min. Distanz(Anchor, Negative)
        ABER Negative != Anchor-Combo
        """
        count = 0
        attempts = 0
        max_attempts = self.num_triplets*20

        while count < self.num_triplets and attempts < max_attempts:
            attempts += 1

            # 1) anchor combo
            anchor_combo = random.choice(self.all_combos)
            anchor_list = self.combo_dict[anchor_combo]
            if len(anchor_list) < 1:
                continue

            # 2) anchor & positive
            anchor_info = random.choice(anchor_list)
            pos_info = random.choice(anchor_list)

            # anchor_index
            a_idx = self.pid_to_index[anchor_info['pid']]

            # 3) Wähle Hard Negative:
            #    => Distmatrix => sort ascending => nimm den index des 1. patients,
            #       der NICHT anchor_combo hat
            dists_a = self.dist_matrix[a_idx]  # shape(N,)

            sorted_indices = torch.argsort(dists_a, dim=0)  # ascending => nearest first

            neg_info = None
            for idx_neg in sorted_indices:
                idx_neg = idx_neg.item()
                if idx_neg == a_idx:
                    continue
                # check combo
                c_neg = self.patient_info_list[idx_neg]['combo']
                if c_neg != anchor_combo:
                    neg_info = self.patient_info_list[idx_neg]
                    break

            if neg_info is None:
                # means all patients in same combo? unwahrscheinlich, skip
                continue

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
