"""
Train Subset Classifiers Script
===============================
Trains all 8 sensor-subset classifiers on the training split,
evaluates on validation and held-out test splits,
and generates the subset comparison table.
"""

import sys
from pathlib import Path
import json
import yaml
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import ParticipantSplitter
from src.feature_extraction import get_feature_column_names
from src.subset_classifiers import SubsetClassifierSuite, SUBSET_DEFINITIONS, SUBSET_NAMES


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    features_path = PROJECT_ROOT / config["data"]["features_path"]
    models_dir = PROJECT_ROOT / config["output"]["models_dir"]
    tables_dir = PROJECT_ROOT / config["output"]["tables_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("STEP 5: TRAINING 8 SENSOR-SUBSET CLASSIFIERS")
    print(f"Features source: {features_path}")
    print("=" * 75)

    if not features_path.exists():
        raise FileNotFoundError(f"Feature table not found: {features_path}. Run 02_prepare_data.py first.")

    df = pd.read_parquet(features_path)
    print(f"Loaded {len(df):,} total windows from {df['subject_id'].nunique()} subjects.")

    # Setup Participant Splitter
    split_cfg = config["splits"]["default"]
    splitter = ParticipantSplitter(
        train_subjects=split_cfg["train"],
        val_subjects=split_cfg["val"],
        test_subjects=split_cfg["test"],
    )
    train_df, val_df, test_df = splitter.split_dataframe(df)

    print(f"\nParticipant Split Summary:")
    print(f"  Train: {splitter.train_subjects} -> {len(train_df):,} windows")
    print(f"  Val:   {splitter.val_subjects} -> {len(val_df):,} windows")
    print(f"  Test:  {splitter.test_subjects} -> {len(test_df):,} windows")

    # Get feature column mapping by sensor group
    group_feats = get_feature_column_names()
    protocol_classes = config["data"]["protocol_activities"]

    suite = SubsetClassifierSuite(
        feature_columns_by_group=group_feats,
        classifier_type=config["models"]["classifier_type"],
        random_state=config["models"]["random_seed"],
    )

    # Train strictly on training data
    suite.fit(train_df, label_col="activity_id", protocol_classes=protocol_classes)

    # Save fitted model suite
    suite.save(models_dir)

    # Evaluate all subsets on train, val, and test
    records = []
    print("\n" + "=" * 85)
    print(f"{'ID':<4} {'Subset Name':<28} {'Split':<8} {'Log Loss':<10} {'Macro-F1':<10} {'Bal Acc':<10} {'Clf Err':<10}")
    print("-" * 85)

    for s_id in range(8):
        s_name = SUBSET_NAMES[s_id]
        eval_train = suite.evaluate_subset(s_id, train_df)
        eval_val = suite.evaluate_subset(s_id, val_df)
        eval_test = suite.evaluate_subset(s_id, test_df)

        for split_name, ev in [("train", eval_train), ("val", eval_val), ("test", eval_test)]:
            records.append({
                "subset_id": s_id,
                "subset_name": s_name,
                "split": split_name,
                "num_sensors": ev["num_sensors"],
                "sensors": ",".join(ev["sensors"]),
                "log_loss": ev["log_loss"],
                "macro_f1": ev["macro_f1"],
                "balanced_accuracy": ev["balanced_accuracy"],
                "classification_error": ev["classification_error"],
            })
            if split_name in ["val", "test"]:
                print(f"{s_id:<4} {s_name:<28} {split_name:<8} {ev['log_loss']:<10.4f} {ev['macro_f1']:<10.4f} {ev['balanced_accuracy']:<10.4f} {ev['classification_error']:<10.4f}")

    eval_df = pd.DataFrame(records)
    out_table = tables_dir / "subset_classifier_performance.csv"
    eval_df.to_csv(out_table, index=False)
    print("=" * 85)
    print(f"\nSaved subset classifier performance table to: {out_table}")

    # Save per-class recall for test split
    per_class_records = []
    for s_id in range(8):
        ev_test = suite.evaluate_subset(s_id, test_df)
        row = {"subset_id": s_id, "subset_name": SUBSET_NAMES[s_id]}
        for cls_id, rec in ev_test["per_class_recall"].items():
            row[f"recall_act_{cls_id}"] = round(rec, 4)
        per_class_records.append(row)

    per_class_df = pd.DataFrame(per_class_records)
    out_per_class = tables_dir / "subset_per_class_recall_test.csv"
    per_class_df.to_csv(out_per_class, index=False)
    print(f"Saved per-class recall table to: {out_per_class}")

    print("\nPhase 5 Training Completed Successfully.")


if __name__ == "__main__":
    main()
