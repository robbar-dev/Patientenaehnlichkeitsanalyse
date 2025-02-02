import random

DEBUG = False

def parse_combo_string(combo_str):
    """
    Für dein neues Binär-Setup ('1-0-0' => (1,), '0-0-1' => (0, ...)),
    damit dein label_distance weiterhin mit Tupeln arbeiten kann.

    => '1-0-0' => (1,)
    => sonst => (0,)  (hier z.B. '0-0-1')

    Wenn du weitere Kombinationen hättest, müsstest du sie
    ebenfalls logisch zu 0/1 zuordnen.
    """
    parts = [int(x) for x in combo_str.split('-')]
    if parts == [1,0,0]:
        return (1,)  # Krank
    else:
        return (0,)  # Gesund

def label_distance(a, b):
    """
    a, b sind nun 1-elementige Tupel, z.B. (0,) oder (1,).
    Die alte Logik summiert Overlap von 1en:
      shared_ones = sum(x & y for x,y in zip(a,b))
    => bei (1,) und (1,) => shared_ones=1 => Distance=-1
    => bei (1,) und (0,) => shared_ones=0 => Distance=0

    So hast du minimal die alte Systematik beibehalten.
    """
    shared_ones = sum(x & y for x,y in zip(a,b))
    return -shared_ones

class TripletSampler:
    """
    Generator für (Anchor, Positive, Negative)
    - Positive = gleicher combo
    - Negative = combos mit minimalem Overlap => label_distance
    - Es wird (label_distance) sortiert => top_k_negatives

    Anders als HardNegativeSampler geht hier alles "label-basiert"
    statt embedding-basiert.
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
            combination z.B. '1-0-0' oder '0-0-1'
        """
        self.df = df.copy()
        self.num_triplets = num_triplets
        self.shuffle = shuffle
        self.top_k_negatives = top_k_negatives

        # 1) Dictionary: combo_str -> [ {pid, study_yr, combo, label}, ... ]
        self.labels_to_patients = {}
        for i, row in self.df.iterrows():
            c_str = row['combination']
            c_tuple = parse_combo_string(c_str)  # z.B. (1,) oder (0,)
            label_val = c_tuple[0]               # 0 oder 1

            if c_str not in self.labels_to_patients:
                self.labels_to_patients[c_str] = []
            self.labels_to_patients[c_str].append({
                'pid': row['pid'],
                'study_yr': row['study_yr'],
                'combination': c_str,
                'label': label_val  # Statt multi_label
            })

        self.all_combos = list(self.labels_to_patients.keys())

        # Shuffle Patient-Listen
        if self.shuffle:
            for c in self.all_combos:
                random.shuffle(self.labels_to_patients[c])

        # 2) Distanz-Ranking der combos (basierend auf label_distance)
        self.distRanking = {}
        for c in self.all_combos:
            c_tuple = parse_combo_string(c)  # (0,) oder (1,)
            combos_sorted = sorted(
                self.all_combos,
                key=lambda other: label_distance(c_tuple, parse_combo_string(other)),
                reverse=True
            )
            # combos_sorted => combos mit minimal Overlap zuerst oder zuletzt
            # je nach Vorzeichen
            self.distRanking[c] = combos_sorted

        # 3) Um doppelte Triplets in einer Epoche zu vermeiden
        self.used_triplets = set()

    def __iter__(self):
        count = 0
        attempts = 0
        max_attempts = self.num_triplets * 20

        while count < self.num_triplets and attempts < max_attempts:
            attempts += 1

            # 1) Anchor combo
            anchor_combo = random.choice(self.all_combos)
            anchor_patients = self.labels_to_patients[anchor_combo]
            if len(anchor_patients) < 1:
                continue

            # 2) Anchor & Positive
            anchor_info = random.choice(anchor_patients)
            pos_info    = random.choice(anchor_patients)

            # 3) Negative combo
            ranking = self.distRanking[anchor_combo]
            # ranking[0] wäre "weitester" oder "engster" je nach dem,
            # wir haben es ja reverse=True => combos mit -sharedOnes "höchster" Dist an erster Stelle
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
