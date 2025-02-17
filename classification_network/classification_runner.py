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

    TRAIN_CSV = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\test_dataset_classification\v1\training\dataset_test_01_training.csv"
    VAL_CSV   = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\test_dataset_classification\v1\val\dataset_test_01_val.csv"
    DATA_ROOT = r"D:\thesis_robert\subset_v5\NLST_subset_v5_nifti_3mm_Voxel"

    # TRAIN_CSV = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\test_dataset_classification\v1\training\dataset_test_01_training.csv"
    # VAL_CSV   = r"C:\Users\rbarbir\OneDrive - Brainlab AG\Dipl_Arbeit\Datensätze\test_dataset_classification\v1\val\dataset_test_01_val.csv"
    # DATA_ROOT = r"D:\thesis_robert\subset_v5\NLST_subset_v5_nifti_1_5mm_Voxel"

    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)

    trainer = ClassificationTrainer(
        df=df_train,
        data_root=DATA_ROOT,
        device='cuda',
        lr=1e-4,
        roi_size=(96, 96, 3),
        overlap=(10, 10, 1),
        skip_factor=2,
        do_augmentation_train=True
    )

    trainer.train_loop(num_epochs=1, df_val=df_val)

    val_loss, val_acc = trainer.eval_on_df(df_val)
    logging.info(f"Final Val => CE-Loss={val_loss:.4f}, Accuracy={val_acc*100:.2f}%")

    trainer.plot_loss_acc()

if __name__=="__main__":
    main()
