import random

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

class BinaryTripletSampler:
    """
    Binäre Klassifikation
    
    - Anchor & Positive kommen aus derselben Klasse
    - Negative kommt aus der jeweils anderen Klasse.
    """

    def __init__(
        self,
        df,
        num_triplets=1000,
        shuffle=True
    ):
        self.df = df.copy()
        self.num_triplets = num_triplets
        self.shuffle = shuffle

        # Dictionary: "0" -> [ {...} ], "1" -> [ {...} ]
        self.labels_to_patients = {"0": [], "1": []}
        for i, row in self.df.iterrows():
            combo_bin = parse_combo_string(row['combination'])  # "0" oder "1"
            ml = [1] if combo_bin == "1" else [0]

            entry = {
                'pid': row['pid'],
                'study_yr': row['study_yr'],
                'combo': combo_bin,      # "0" oder "1"
                'multi_label': ml        # [0] oder [1]
            }
            self.labels_to_patients[combo_bin].append(entry)

        self.all_labels = ["0", "1"]

        if self.shuffle:
            for lb in self.all_labels:
                random.shuffle(self.labels_to_patients[lb])

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

            anchor_label = random.choice(self.all_labels)
            anchor_patients = self.labels_to_patients[anchor_label]
            if len(anchor_patients) < 2:
                continue

            anchor_info = random.choice(anchor_patients)
            positive_info = random.choice(anchor_patients)

            negative_label = "1" if anchor_label == "0" else "0"
            neg_patients = self.labels_to_patients[negative_label]
            if len(neg_patients) < 1:
                continue

            negative_info = random.choice(neg_patients)

            # Triplet-ID, um Duplikate zu verhindern
            trip_id = (
                anchor_info['pid'],
                positive_info['pid'],
                negative_info['pid'],
                anchor_label,
                negative_label
            )
            if trip_id in self.used_triplets:
                continue

            self.used_triplets.add(trip_id)
            yield (anchor_info, positive_info, negative_info)
            count += 1

        if DEBUG:
            print(f"[BinaryTripletSampler] Generated {count} triplets out of {attempts} attempts.")

    def __len__(self):
        return self.num_triplets

    def reset_epoch(self):
        self.used_triplets = set()
