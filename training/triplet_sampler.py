import random

DEBUG = False

def parse_combo_string(combo_str):
    """
    '0-1-1' -> (0,1,1)
    """
    return tuple(int(x) for x in combo_str.split('-'))

def label_distance(a, b):
    """
    a, b = Tupel, z. B. (0,1,1)
    Distance = -#gemeinsamer1en
    => je weniger Overlap, desto größer die Distance
    """
    shared_ones = sum(x & y for x,y in zip(a,b))
    return -shared_ones

class TripletSampler:
    """
    Generator für (Anchor, Positive, Negative),
    inkl. Multi-Label-Vektor in anchor_info['multi_label'] etc.
    """

    def __init__(
        self,
        df,
        num_triplets=1000,
        shuffle=True,
        top_k_negatives=5
    ):
        """
        df: Pandas DataFrame mit Spalten [pid, study_yr, combination]
            combination z.B. '0-1-1'
        """
        self.df = df.copy()
        self.num_triplets = num_triplets
        self.shuffle = shuffle
        self.top_k_negatives = top_k_negatives

        # 1) Dictionary: combo_str -> [ {pid, study_yr, combo, multi_label}, ... ]
        self.labels_to_patients = {}
        for i, row in self.df.iterrows():
            c_str = row['combination']       # z.B. '0-1-1'
            c_tuple = parse_combo_string(c_str)  # (0,1,1)
            ml_vec = list(c_tuple)               # [0,1,1]
            if c_str not in self.labels_to_patients:
                self.labels_to_patients[c_str] = []
            self.labels_to_patients[c_str].append({
                'pid': row['pid'],
                'study_yr': row['study_yr'],
                'combination': c_str,
                'multi_label': ml_vec  # <--- WICHTIG
            })

        self.all_combos = list(self.labels_to_patients.keys())

        # Shuffle Patient-Listen
        if self.shuffle:
            for c in self.all_combos:
                random.shuffle(self.labels_to_patients[c])

        # 2) Distanz-Ranking
        self.distRanking = {}
        for c in self.all_combos:
            c_tuple = parse_combo_string(c)
            combos_sorted = sorted(
                self.all_combos,
                key=lambda other: label_distance(c_tuple, parse_combo_string(other)),
                reverse=True
            )
            self.distRanking[c] = combos_sorted

        # 3) "Ohne Zurücklegen" => set()
        self.used_triplets = set()

    def __iter__(self):
        count = 0
        attempts = 0
        max_attempts = self.num_triplets * 20

        while count < self.num_triplets and attempts < max_attempts:
            attempts += 1

            # 1) anchor combo
            anchor_combo = random.choice(self.all_combos)
            anchor_patients = self.labels_to_patients[anchor_combo]
            if len(anchor_patients) < 1:
                continue

            # 2) anchor & positive
            anchor_info = random.choice(anchor_patients)
            pos_info    = random.choice(anchor_patients)

            # 3) neg combo
            ranking = self.distRanking[anchor_combo]
            ranking_filtered = [rc for rc in ranking if rc != anchor_combo]
            if not ranking_filtered:
                continue
            top_k = ranking_filtered[:self.top_k_negatives]
            neg_combo = random.choice(top_k)
            neg_patients = self.labels_to_patients[neg_combo]
            if len(neg_patients) < 1:
                continue
            neg_info = random.choice(neg_patients)

            # Triplet ID
            trip_id = (
                anchor_info['pid'],
                pos_info['pid'],
                neg_info['pid'],
                anchor_combo,
                neg_combo
            )

            if trip_id in self.used_triplets:
                continue

            self.used_triplets.add(trip_id)
            yield (anchor_info, pos_info, neg_info)
            count += 1

        if DEBUG:
            print(f"[TripletSampler] Generated {count} triplets out of {attempts} attempts.")

    def __len__(self):
        return self.num_triplets

    def reset_epoch(self):
        self.used_triplets = set()
