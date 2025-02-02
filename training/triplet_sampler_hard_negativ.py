import random
import torch

DEBUG = False

def parse_combo_string(combo_str):
    """
    Für deine Binär-Klassen ('1-0-0' => krank, '0-0-1' => gesund)
    erzeugen wir ein eindimensionales Tuple:
      '1-0-0' => (1,)
      alles andere => (0,)
    Damit bleibt das Format Tuple[int] erhalten, und z.B. der Sampler-Code
    kann intern weiterhin combos unterscheiden.
    """
    parts = [int(x) for x in combo_str.split('-')]
    if parts == [1,0,0]:
        return (1,)
    else:
        # Falls du nur 0-0-1 im Datensatz hast, definieren wir den Default als (0,)
        return (0,)

class HardNegativeTripletSampler:
    """
    - Offline-Berechnung der Embeddings für alle Patienten via 'trainer.compute_patient_embedding'
    - Pro Anchor wählt man:
        Anchor => (pid, study_yr, combo, label)
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
        #    Wir speichern in patient_info: {
        #      'pid','study_yr','combo','label','embedding'
        #    }
        print("[HardNegSampler] Compute Embeddings for all patients ...")
        self.patient_info_list = []
        with torch.no_grad():
            for i, row in self.df.iterrows():
                pid = row['pid']
                sy  = row['study_yr']
                combo = row['combination']

                # parse combo => z.B. (1,) oder (0,)
                combo_tuple = parse_combo_string(combo)
                label_val   = combo_tuple[0]  # 0 oder 1

                emb = trainer.compute_patient_embedding(pid, sy)
                # emb shape (1, d)
                emb = emb.squeeze(0).cpu()

                self.patient_info_list.append({
                    'pid': pid,
                    'study_yr': sy,
                    'combo': combo,          # originaler String "1-0-0" oder "0-0-1"
                    'label': label_val,      # neu: nur 0/1
                    'embedding': emb
                })

        # 2) combo_dict: combo_str -> Liste von patient_info
        self.combo_dict = {}
        for pinfo in self.patient_info_list:
            c = pinfo['combo']
            if c not in self.combo_dict:
                self.combo_dict[c] = []
            self.combo_dict[c].append(pinfo)

        self.all_combos = list(self.combo_dict.keys())

        # 3) Alle Embeddings in ein Tensor (N,d) => Distanzmatrix
        self.emb_tensor = torch.stack([p['embedding'] for p in self.patient_info_list], dim=0)
        # => shape (N,d)

        # Index-Mapping (pid->Index) und umgekehrt
        self.pid_to_index = {}
        for idx, pinfo in enumerate(self.patient_info_list):
            self.pid_to_index[pinfo['pid']] = idx

        # 4) Distanzmatrix (N,N)
        self.dist_matrix = torch.cdist(
            self.emb_tensor.unsqueeze(0),
            self.emb_tensor.unsqueeze(0),
            p=2
        ).squeeze(0)
        # => shape (N,N)

        # 5) used_triplets => pro Epoche
        self.used_triplets = set()

    def __iter__(self):
        """
        generiert num_triplets Triplets:
          Anchor + Positive (gleiche combo)
          Hard Negative = kleinste embedding-dist, aber andere combo
        """
        count = 0
        attempts = 0
        max_attempts = self.num_triplets * 20

        while count < self.num_triplets and attempts < max_attempts:
            attempts += 1

            # 1) Wähle random anchor combo
            anchor_combo = random.choice(self.all_combos)
            anchor_list = self.combo_dict[anchor_combo]
            if len(anchor_list) < 1:
                continue

            # 2) anchor & positive
            anchor_info = random.choice(anchor_list)
            pos_info    = random.choice(anchor_list)

            # 3) Hard Negative
            a_idx = self.pid_to_index[anchor_info['pid']]
            dists_a = self.dist_matrix[a_idx]  # shape(N,)

            # sort ascending => nearest first
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
                # Falls man keinen neg findet (unwahrscheinlich) -> skip
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
            print(f"[HardNegativeTripletSampler] Generated {count} triplets out of {attempts} attempts.")

    def __len__(self):
        return self.num_triplets

    def reset_epoch(self):
        self.used_triplets = set()
