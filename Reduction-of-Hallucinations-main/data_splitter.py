import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def split_raw_data(
    input_file="Sets/medhal_preprocessed.csv",
    output_dir="Sets",
    max_rows=10000,
    random_state=42
):
    print(f"Loading raw data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Shuffle initially to remove any ordering bias
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    orig_n = len(df)
    print(f"Original rows: {orig_n}")

    # Decide stratify column if present
    stratify_col = "domain" if "domain" in df.columns else None

    # If dataset is larger than max_rows, sample down to max_rows,
    # preserving stratification if possible.
    if orig_n > max_rows:
        print(f"Dataset larger than {max_rows}. Sampling down to {max_rows} rows...")
        if stratify_col:
            # Attempt stratified sampling using StratifiedShuffleSplit
            try:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=max_rows, random_state=random_state)
                split_idx = next(sss.split(df, df[stratify_col]))
                subset_df = df.iloc[split_idx[0]].reset_index(drop=True)
            except Exception as e:
                # Fall back to random sampling if stratified sampling fails (e.g., too few samples per class)
                print(f"Warning: stratified sampling failed ({e}). Falling back to random sampling.")
                subset_df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        else:
            subset_df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
    else:
        subset_df = df.copy()
        if orig_n < max_rows:
            print(f"Dataset has only {orig_n} rows (< {max_rows}). Using all rows.")

    n = len(subset_df)
    print(f"Using {n} rows for split (max_rows={max_rows}).")

    # First split: test (15%) and remaining (85%)
    try:
        train_val_df, test_df = train_test_split(
            subset_df,
            test_size=0.15,
            random_state=random_state,
            stratify=subset_df[stratify_col] if stratify_col else None
        )
    except ValueError as e:
        # Could happen if stratify fails (e.g., too few samples in some classes)
        print(f"Warning: initial stratified split failed ({e}). Retrying without stratification.")
        train_val_df, test_df = train_test_split(
            subset_df,
            test_size=0.15,
            random_state=random_state,
            stratify=None
        )

    # Second split: validation is 10% of the training set (i.e., 10% of 85% = 8.5% overall)
    try:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.10,  # 10% of the training set
            random_state=random_state,
            stratify=train_val_df[stratify_col] if stratify_col else None
        )
    except ValueError as e:
        print(f"Warning: stratified train/val split failed ({e}). Retrying without stratification.")
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=0.10,
            random_state=random_state,
            stratify=None
        )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save files
    train_path = os.path.join(output_dir, "train_set.csv")
    val_path = os.path.join(output_dir, "validation_set.csv")
    test_path = os.path.join(output_dir, "test_set.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("✓ Splitting Complete:")
    print(f"  - Train: {len(train_df)} rows (for SFT & DPO) -> {train_path}")
    print(f"  - Val:   {len(val_df)} rows (for training checks) -> {val_path}")
    print(f"  - Test:  {len(test_df)} rows (LOCKED for final eval) -> {test_path}")

    # Report proportions
    print("\nProportions (approx):")
    print(f"  - Train: {len(train_df)} / {n} = {len(train_df)/n:.3f}")
    print(f"  - Val:   {len(val_df)} / {n} = {len(val_df)/n:.3f}")
    print(f"  - Test:  {len(test_df)} / {n} = {len(test_df)/n:.3f}")

if __name__ == "__main__":
    split_raw_data()