import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_dataset(
    file_path='namadataset_raw/StudentsPerformance.csv',
    test_size=0.2,
    random_state=42,
    output_dir='preprocessing'
):
    # ===============================
    # 1. Load Dataset
    # ===============================
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully.")
    except Exception as e:
        print("❌ Failed to load dataset:", e)
        return None, None, None, None

    # ===============================
    # 2. Rename Columns (snake_case)
    # ===============================
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    print("📋 Kolom dalam dataset:", df.columns.tolist())

    # ===============================
    # 3. Missing Values Handling
    # ===============================
    if df.isnull().sum().sum() > 0:
        df.dropna(inplace=True)
        print("✔ Missing values removed.")

    # ===============================
    # 4. Drop Duplicates
    # ===============================
    df.drop_duplicates(inplace=True)

    # ===============================
    # 5. Backup Original Numeric Scores
    # ===============================
    numeric_cols = ['math_score', 'reading_score', 'writing_score']
    for col in numeric_cols:
        if col in df.columns:
            df[col + '_original'] = df[col]

    # ===============================
    # 6. Outlier Removal (IQR)
    # ===============================
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    # ===============================
    # 7. Label Encoding
    # ===============================
    expected_label_cols = ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
    label_cols = [col for col in expected_label_cols if col in df.columns]

    if label_cols:
        le = LabelEncoder()
        for col in label_cols:
            df[col + '_original'] = df[col]
            df[col] = le.fit_transform(df[col])
        print(f"🔤 Label encoding applied to: {label_cols}")
    else:
        print("⚠️ No label columns found for encoding.")

    # ===============================
    # 8. Normalisasi
    # ===============================
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("📊 Normalisasi kolom numerik selesai.")

    # ===============================
    # 9. Binning Target Variable
    # ===============================
    def bin_math(score_std):
        mean = df['math_score_original'].mean()
        std = df['math_score_original'].std()
        actual = score_std * std + mean
        if actual <= 60:
            return 0
        elif actual <= 80:
            return 1
        else:
            return 2

    df['math_score_binned'] = df['math_score'].apply(bin_math)

    # ===============================
    # 10. Save Cleaned Dataset Automatically
    # ===============================
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'StudentsPerformance_cleaned.csv')
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned dataset automatically saved to: {output_path}")

    # ===============================
    # 11. Split Data
    # ===============================
    X = df.drop(columns=['math_score_binned', 'math_score_original', 'reading_score_original', 'writing_score_original'], errors='ignore')
    y = df['math_score_binned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"📤 Data split completed: {X_train.shape[0]} train / {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test


# Contoh penggunaan
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_dataset()
