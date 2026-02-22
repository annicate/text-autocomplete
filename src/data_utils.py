import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    df["text"] = df["text"].astype(str).apply(clean_text)
    df.to_csv(output_path, index=False)


def split_dataset(processed_path, output_dir):
    df = pd.read_csv(processed_path)
    
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42
    )
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)