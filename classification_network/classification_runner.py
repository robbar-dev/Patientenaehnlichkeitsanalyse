import os
import sys
import logging
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from classification_trainer import ClassificationTrainer

def main():
    logging.basicConfig(level=logging.INFO)

    TRAIN_CSV = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\training\nlst_subset_v5_training.csv"
    VAL_CSV   = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\Subsets\V5\validation\nlst_subset_v5_validation.csv"
    DATA_ROOT = r"D:\thesis_robert\NLST_subset_v5_nifti_3mm_Voxel"

    # df laden => filtern combos => 3 klassen
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)

    trainer = ClassificationTrainer(
        df=df_train,
        data_root=DATA_ROOT,
        device='cuda',
        lr=1e-4,
        roi_size=(96,96,3),
        overlap=(10,10,1),
        skip_slices=True,
        skip_factor=2,
        filter_empty_patches=False,
        filter_uniform_patches=False,
        do_augmentation_train=True
    )

    # 1) train
    trainer.train_loop(num_epochs=5) 

    # 2) Evaluate => on val
    val_df = df_val.copy()
    # Ggf. filter combos => 1-0-0,0-1-0,0-0-1
    valid_combos = {"1-0-0","0-1-0","0-0-1"}
    val_df = val_df[val_df["combination"].isin(valid_combos)].copy()
    val_df["cls_label"] = val_df["combination"].apply(lambda c: 0 if c=="1-0-0" else (1 if c=="0-1-0" else (2 if c=="0-0-1" else None)))
    val_df = val_df.dropna(subset=["cls_label"])
    val_df["cls_label"] = val_df["cls_label"].astype(int)

    val_loss, val_acc = trainer.eval_on_df(val_df)
    logging.info(f"Val => CE-Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")

    # 3) optional => plot
    trainer.plot_loss_acc()

if __name__=="__main__":
    main()