"""
=============================================================================
preprocess.py - Dataset Preprocessing Module
=============================================================================
Project  : Sign Language Recognition
Python   : 3.11.9

Description:
    Loads the raw CSV dataset collected by collect_data.py, validates it,
    handles missing values, encodes labels using LabelEncoder, and splits
    the dataset into 80% training and 20% testing sets.
    
    The preprocessed arrays are saved for the training module to use,
    and the LabelEncoder is saved for inference.
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Constants
CSV_FILE = os.path.join("dataset", "raw", "sign_landmarks.csv")
PROCESSED_DIR = os.path.join("dataset", "processed")
MODELS_DIR = "models"
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

def main():
    print("=" * 60)
    print("   Sign Language Recognition - Preprocessing")
    print("=" * 60)
    
    # 1. Validate paths and create necessary directories
    if not os.path.exists(CSV_FILE):
        print(f"[-] Error: Raw dataset not found at {CSV_FILE}")
        return
        
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 2. Load CSV dataset
    print(f"[*] Loading dataset from {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"[-] Error loading CSV: {e}")
        return
        
    print(f"[+] Dataset loaded successfully. Shape: {df.shape}")
    
    # 3. Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"[!] Warning: Found {missing_values} missing values in the dataset.")
        print("[*] Dropping rows with missing values...")
        df = df.dropna()
        print(f"[+] Cleaned dataset shape: {df.shape}")
    else:
        print("[+] No missing values detected.")
        
    # 4. Validate data structure (label + 63 feature columns)
    if df.shape[1] != 64:
        print(f"[-] Error: Expected 64 columns (1 label + 63 features), but got {df.shape[1]}")
        return
        
    # 5. Separate features (X) and labels (y)
    print("[*] Separating features and labels...")
    X = df.drop("label", axis=1).values
    y = df["label"].astype(str).values  # Ensure labels are treated as strings (e.g., '0' instead of 0)
    
    # 6. Encode labels using LabelEncoder
    print("[*] Encoding labels...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Save the encoder for future use (training and inference)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"[+] LabelEncoder saved to {ENCODER_PATH}")
    print(f"    - Classes found: {encoder.classes_}")
    
    # 7. Split dataset into 80% train and 20% test
    print("[*] Splitting dataset into 80% train and 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.20, 
        random_state=42, 
        stratify=y_encoded  # Ensures balanced class distribution in train/test splits
    )
    
    print(f"    - Training samples: {X_train.shape[0]}")
    print(f"    - Testing samples : {X_test.shape[0]}")
    
    # 8. Save preprocessed data
    print("[*] Saving preprocessed arrays...")
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
    
    print(f"[+] Preprocessed data saved in {PROCESSED_DIR}")
    print("=" * 60)
    print(">>> Preprocessing complete. Ready for model training.")
    print("=" * 60)

if __name__ == "__main__":
    main()
