# utils/data_balancing.py
# ---------------------------------------------------------------------------- #
# Data balancing UI + logic
# ---------------------------------------------------------------------------- #

import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------- #
@st.cache_data
def load_and_balance_data(file_path, method, target_count, allow_duplicates=False):
    """
    Load CSV, map Toxic_flag to booleans, deduplicate unique Text, and balance.

    Args:
      file_path: path to CSV
      method: "Undersampling" or "Oversampling"
      target_count: desired per-class sample count (int)
      allow_duplicates: if True and method=="Oversampling" -> resample with replacement (classic oversample).
                        if False and method=="Oversampling" -> do NOT create duplicates; majority may be undersampled.
    """
    df = pd.read_csv(file_path)

    if len(df) > 200000:
        # larger threshold but we warn
        st.warning("⚠️ Large dataset detected. Sampling 100,000 rows to improve interactivity.")
        df = df.sample(n=100000, random_state=42)

    if 'Toxic_flag' not in df.columns:
        st.error("❌ The dataset must contain a `Toxic_flag` column.")
        return None

    if df['Toxic_flag'].isna().any():
        st.warning("⚠️ Missing values in `Toxic_flag`. Dropping rows with missing values.")
        df = df.dropna(subset=['Toxic_flag'])

    def map_flag(x):
        if isinstance(x, bool):
            return x
        s = str(x).strip()
        if s in {'1', 'True', 'true', 'TRUE', 'T', 't', 'yes', 'Yes', 'Y', 'y'}:
            return True
        if s in {'0', 'False', 'false', 'FALSE', 'F', 'f', 'no', 'No', 'N', 'n'}:
            return False
        if 'tox' in s.lower():
            return True
        return bool(s)

    df['Toxic_flag'] = df['Toxic_flag'].apply(map_flag)

    if 'Text' not in df.columns:
        st.error("❌ The dataset must have a 'Text' column.")
        return None

    before = len(df)
    df = df.drop_duplicates(subset=['Text']).reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        st.info(f"🧹 Removed {removed} duplicate text entries before balancing.")

    balanced_df = balance_binary(df, method, int(target_count), allow_duplicates)

    if balanced_df is None or balanced_df.empty:
        st.error("❌ Balancing failed: No data available after processing.")
        return None

    return balanced_df[['Text', 'Toxic_flag']]


def render_data_balancing_ui():
    st.title("⚖️ Data Balancing")
    st.markdown("### Advanced preprocessing and dataset balancing operations")
    st.markdown("---")

    st.caption("📂 Select dataset to balance:")
    file_path = st.text_input(
        "Enter path to your CSV file:",
        r"C:\Users\jayav\Downloads\Save\College\Capstone\cyberbullying-telugu\data\toxic_data_cleaned_large.csv"
    )

    if not file_path or not os.path.exists(file_path):
        st.error("❌ File not found. Please check path.")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return

    st.subheader("Current Class Distribution")
    plot_class_distribution(df)

    st.markdown("---")
    st.subheader("Balancing Options")
    method = st.selectbox("Method:", ["Undersampling", "Oversampling"])
    target_count = st.number_input("Target Samples per Class:", min_value=100, max_value=100000, value=1000, step=100)

    allow_duplicates = False
    if method == "Oversampling":
        allow_duplicates = st.checkbox("Allow duplicates when oversampling (replace=True)", value=False,
                                       help="If unchecked, oversampling will NOT create duplicates; majority class will be undersampled instead to keep samples unique.")

    save_path = "data/training/binary/dataset_binary.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if st.button("Preview Balanced Data"):
        balanced_df = load_and_balance_data(file_path, method, target_count, allow_duplicates)
        if balanced_df is not None:
            st.session_state['balanced_df'] = balanced_df
            st.session_state['save_path'] = save_path

    if 'balanced_df' in st.session_state:
        st.subheader("✅ Balanced Distribution Preview")
        plot_class_distribution(st.session_state['balanced_df'])
        st.dataframe(st.session_state['balanced_df'].head(10))

        if st.button("Save Balanced Dataset"):
            try:
                st.session_state['balanced_df'].to_csv(st.session_state['save_path'], index=False)
                st.success(f"✅ Saved to: `{st.session_state['save_path']}`")
            except Exception as e:
                st.error(f"❌ Failed to save file: {str(e)}")

    # -------------------------------------------------------------------------
    # New Section: Create train/val/test splits from the balanced dataset
    # Non-invasive: only shows if user has balanced_df OR explicit selection
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.header("🚦 Create train/val/test splits (from balanced dataset)")

    # Choose source dataframe: prefer last balanced preview, else allow explicit selection
    source_option = "Use previewed balanced dataset" if 'balanced_df' in st.session_state else "Load dataset file path"
    source_choice = st.radio("Source for split creation:", [source_option, "Load dataset file path"], index=0)

    if source_choice == source_option and 'balanced_df' in st.session_state:
        src_df = st.session_state['balanced_df'].copy()
        st.info("Using the balanced dataframe you previewed above.")
    else:
        # allow loading a different CSV explicitly
        split_file = st.text_input("Enter dataset CSV path for splitting (must have 'Text' and 'Toxic_flag'):", file_path)
        if not split_file or not os.path.exists(split_file):
            st.warning("Provide a valid CSV path to enable split creation.")
            src_df = None
        else:
            try:
                src_df = pd.read_csv(split_file)
            except Exception as e:
                st.error(f"Failed to load split CSV: {e}")
                src_df = None

    if src_df is None:
        st.info("No dataset available for split creation yet.")
        return

    st.subheader("Split Settings")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        train_frac = st.slider("Train fraction", 0.5, 0.9, 0.8, 0.05)
    with col_b:
        val_frac = st.slider("Validation fraction", 0.05, 0.3, 0.1, 0.05)
    # derive test fraction
    test_frac = max(0.0, 1.0 - train_frac - val_frac)
    col_c.markdown(f"**Test fraction (auto):** {test_frac:.2f}")

    random_state = st.number_input("Random seed:", min_value=0, max_value=99999, value=42, step=1)

    dedupe_check = st.checkbox("Check overlaps between splits before saving (recommended)", value=True)
    show_overlap_button = st.checkbox("Show overlapping rows (if any) before creating files", value=False)

    out_dir = "data/training/splits"
    st.markdown(f"**Output folder:** `{out_dir}`")
    os.makedirs(out_dir, exist_ok=True)

    # Preview split button
    if st.button("▶️ Create Splits (preview only)"):
        # Prepare df: ensure required columns
        if 'Text' not in src_df.columns or 'Toxic_flag' not in src_df.columns:
            st.error("Dataset must contain 'Text' and 'Toxic_flag' columns to create stratified splits.")
        else:
            # drop NA text and labels
            df_clean = src_df.dropna(subset=['Text', 'Toxic_flag']).reset_index(drop=True).copy()
            # normalize label to 0/1
            df_clean['label'] = df_clean['Toxic_flag'].apply(lambda x: 1 if bool(x) else 0)

            # sample cap to avoid OOM in UI (but still allow full save option later)
            max_preview = 20000
            if len(df_clean) > max_preview:
                st.warning(f"Large dataset ({len(df_clean)} rows). Sampling {max_preview} rows for preview split only.")
                df_sample = df_clean.sample(n=max_preview, random_state=random_state).reset_index(drop=True)
            else:
                df_sample = df_clean

            # Stratified split if label exists with at least 2 classes
            ratios_valid = (train_frac + val_frac) < 1.0
            if not ratios_valid:
                st.error("Train + Validation fractions must be < 1.0")
            else:
                try:
                    # first split off train
                    temp_frac = train_frac
                    rest_frac = 1.0 - temp_frac
                    # compute relative val size w.r.t rest
                    if rest_frac <= 0:
                        st.error("Invalid fractions (rest <= 0). Adjust sliders.")
                    else:
                        val_rel = val_frac / rest_frac
                        train_df, temp_df = train_test_split(
                            df_sample,
                            train_size=temp_frac,
                            stratify=df_sample['label'],
                            random_state=random_state
                        )
                        val_df, test_df = train_test_split(
                            temp_df,
                            train_size=val_rel,
                            stratify=temp_df['label'],
                            random_state=random_state
                        )

                        st.success("Preview splits created.")
                        st.write(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

                        # optional overlap check
                        if dedupe_check:
                            overlap_train_val = pd.merge(train_df[['Text']], val_df[['Text']], on='Text', how='inner')
                            overlap_train_test = pd.merge(train_df[['Text']], test_df[['Text']], on='Text', how='inner')
                            overlap_val_test = pd.merge(val_df[['Text']], test_df[['Text']], on='Text', how='inner')
                            total_overlaps = len(overlap_train_val) + len(overlap_train_test) + len(overlap_val_test)
                            if total_overlaps > 0:
                                st.warning(f"Detected {total_overlaps} overlapping 'Text' rows across splits.")
                                if show_overlap_button:
                                    st.subheader("Overlaps: Train vs Val")
                                    st.dataframe(overlap_train_val.head(30))
                                    st.subheader("Overlaps: Train vs Test")
                                    st.dataframe(overlap_train_test.head(30))
                                    st.subheader("Overlaps: Val vs Test")
                                    st.dataframe(overlap_val_test.head(30))
                            else:
                                st.success("No overlapping 'Text' rows detected across splits (good).")

                        # show counts by label
                        st.markdown("Label distribution (train / val / test)")
                        dist_df = pd.DataFrame({
                            "train": train_df['label'].value_counts().sort_index(),
                            "val": val_df['label'].value_counts().sort_index(),
                            "test": test_df['label'].value_counts().sort_index()
                        }).fillna(0).astype(int)
                        st.dataframe(dist_df)

                        # keep preview in session for final save
                        st.session_state['preview_splits'] = {
                            "train": train_df,
                            "val": val_df,
                            "test": test_df
                        }
                except Exception as e:
                    st.error(f"Could not create splits: {e}")

    # Final save button: write full splits (use entire src_df, not preview sample)
    if st.button("💾 Save Splits to disk (write full dataset)"):
        # validate columns
        if 'Text' not in src_df.columns or 'Toxic_flag' not in src_df.columns:
            st.error("Dataset must contain 'Text' and 'Toxic_flag' columns to create stratified splits.")
        else:
            df_clean = src_df.dropna(subset=['Text', 'Toxic_flag']).reset_index(drop=True).copy()
            df_clean['label'] = df_clean['Toxic_flag'].apply(lambda x: 1 if bool(x) else 0)

            # produce full splits now (may be large)
            try:
                # create splits deterministically using train_test_split with stratify
                temp_frac = train_frac
                rest_frac = 1.0 - temp_frac
                if rest_frac <= 0:
                    st.error("Invalid fractions (rest <= 0). Adjust sliders.")
                else:
                    val_rel = val_frac / rest_frac
                    train_df, temp_df = train_test_split(
                        df_clean,
                        train_size=temp_frac,
                        stratify=df_clean['label'],
                        random_state=random_state
                    )
                    val_df, test_df = train_test_split(
                        temp_df,
                        train_size=val_rel,
                        stratify=temp_df['label'],
                        random_state=random_state
                    )

                    # optional dedupe overlap check
                    overlap_info = {}
                    if dedupe_check:
                        overlap_train_val = pd.merge(train_df[['Text']], val_df[['Text']], on='Text', how='inner')
                        overlap_train_test = pd.merge(train_df[['Text']], test_df[['Text']], on='Text', how='inner')
                        overlap_val_test = pd.merge(val_df[['Text']], test_df[['Text']], on='Text', how='inner')
                        overlap_info['train_val'] = len(overlap_train_val)
                        overlap_info['train_test'] = len(overlap_train_test)
                        overlap_info['val_test'] = len(overlap_val_test)
                        total_overlaps = sum(overlap_info.values())
                        if total_overlaps > 0:
                            st.warning(f"Detected overlaps across splits: {overlap_info}. You may want to inspect before training.")
                            if show_overlap_button:
                                st.subheader("Overlaps: Train vs Val (sample)")
                                st.dataframe(overlap_train_val.head(100))
                                st.subheader("Overlaps: Train vs Test (sample)")
                                st.dataframe(overlap_train_test.head(100))
                                st.subheader("Overlaps: Val vs Test (sample)")
                                st.dataframe(overlap_val_test.head(100))
                        else:
                            st.success("No overlaps detected across full splits.")

                    # write CSVs
                    train_path = os.path.join(out_dir, "train.csv")
                    val_path = os.path.join(out_dir, "val.csv")
                    test_path = os.path.join(out_dir, "test.csv")
                    try:
                        train_df[['Text', 'Toxic_flag']].to_csv(train_path, index=False)
                        val_df[['Text', 'Toxic_flag']].to_csv(val_path, index=False)
                        test_df[['Text', 'Toxic_flag']].to_csv(test_path, index=False)
                        st.success(f"✅ Splits saved: `{train_path}`, `{val_path}`, `{test_path}`")
                        # save a metadata file
                        meta = {
                            "train_rows": len(train_df),
                            "val_rows": len(val_df),
                            "test_rows": len(test_df),
                            "train_frac": train_frac,
                            "val_frac": val_frac,
                            "test_frac": test_frac,
                            "random_state": random_state,
                            "overlap_info": overlap_info
                        }
                        try:
                            import json
                            with open(os.path.join(out_dir, "split_metadata.json"), "w", encoding="utf-8") as f:
                                json.dump(meta, f, indent=2)
                        except Exception:
                            pass
                    except Exception as e:
                        st.error(f"Failed to write split files: {e}")
            except Exception as e:
                st.error(f"Could not create & save splits: {e}")

# ---------------------------------------------------------------------------- #
def plot_class_distribution(df):
    if df is None or df.empty:
        st.error("❌ Cannot plot: Dataset is empty.")
        return
    if 'Toxic_flag' not in df.columns:
        st.error("❌ Cannot plot: `Toxic_flag` column missing.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['Toxic_flag'].value_counts().rename({True: 'Toxic', False: 'Non-Toxic'})
    counts = counts.reindex(['Non-Toxic', 'Toxic']).fillna(0)
    counts.plot(kind='bar', ax=ax)
    ax.set_title("Toxic vs Non-Toxic Distribution")
    plt.xticks(rotation=0, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def balance_binary(df, method, target_count, allow_duplicates=False):
    """
    Balanced dataframe with choice to allow duplicates for oversampling.

    If allow_duplicates==False and method=="Oversampling", we avoid creating duplicates:
      - we will sample the majority down to minority size (i.e., produce balanced unique-set).
    If allow_duplicates==True and method=="Oversampling", we will resample minority with replacement to reach target_count.
    """

    toxic = df[df['Toxic_flag'] == True]
    non_toxic = df[df['Toxic_flag'] == False]

    if len(toxic) == 0 or len(non_toxic) == 0:
        st.error("❌ Cannot balance: One or both classes have no samples.")
        return pd.DataFrame(columns=df.columns)

    toxic_count = len(toxic)
    non_toxic_count = len(non_toxic)

    if method == "Undersampling":
        desired = min(toxic_count, non_toxic_count, int(target_count))
        st.info(f"🔽 Undersampling both classes to {desired} samples each (no duplicates).")
        toxic_bal = resample(toxic, replace=False, n_samples=desired, random_state=42)
        non_toxic_bal = resample(non_toxic, replace=False, n_samples=desired, random_state=42)

    else:  # Oversampling
        if allow_duplicates:
            # classic oversample with replacement to target_count
            desired = int(target_count)
            desired = max(desired, min(toxic_count, non_toxic_count))  # at least minority size
            st.info(f"🔼 Oversampling with replacement to {desired} samples each (duplicates allowed).")
            toxic_bal = resample(toxic, replace=True, n_samples=desired, random_state=42)
            non_toxic_bal = resample(non_toxic, replace=True, n_samples=desired, random_state=42)
        else:
            # do NOT create duplicates - instead produce unique balanced set by sampling majority down
            st.info("⚠️ Oversampling without duplicates: reducing majority class to minority size to keep unique texts.")
            if toxic_count < non_toxic_count:
                desired = toxic_count
                toxic_bal = toxic
                non_toxic_bal = resample(non_toxic, replace=False, n_samples=desired, random_state=42)
            elif non_toxic_count < toxic_count:
                desired = non_toxic_count
                non_toxic_bal = non_toxic
                toxic_bal = resample(toxic, replace=False, n_samples=desired, random_state=42)
            else:
                # already balanced
                toxic_bal = toxic
                non_toxic_bal = non_toxic

    balanced = pd.concat([toxic_bal, non_toxic_bal], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced
