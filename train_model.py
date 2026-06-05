"""
=============================================================================
train_model.py - Model Training and Evaluation Module
=============================================================================
Project  : Sign Language Recognition
Python   : 3.11.9

Description:
    Loads the preprocessed dataset, trains a Random Forest Classifier,
    evaluates its performance on training and testing sets, prints detailed
    metrics (accuracy, classification report, confusion matrix), and saves
    the trained model and visual evaluation plots.
=============================================================================
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Constants
PROCESSED_DIR = os.path.join("dataset", "processed")
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
PLOTS_DIR = "evaluation"

def main():
    print("=" * 60)
    print("   Sign Language Recognition - Model Training")
    print("=" * 60)
    
    # 1. Validate data existence
    data_files = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]
    for file in data_files:
        if not os.path.exists(os.path.join(PROCESSED_DIR, file)):
            print(f"[-] Error: Preprocessed data missing: {file}. Run preprocess.py first.")
            return
            
    if not os.path.exists(ENCODER_PATH):
        print(f"[-] Error: Label encoder missing at {ENCODER_PATH}. Run preprocess.py first.")
        return
        
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 2. Load preprocessed data
    print("[*] Loading preprocessed data...")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    
    # Load encoder to get class names for reports
    encoder = joblib.load(ENCODER_PATH)
    class_names = encoder.classes_
    
    print(f"    - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"    - X_test shape : {X_test.shape}, y_test shape : {y_test.shape}")
    
    # 3. Train Random Forest Classifier
    print("\n[*] Initializing and training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Fit the model
    model.fit(X_train, y_train)
    print("[+] Model training completed.")
    
    # 4. Model Evaluation
    print("\n[*] Evaluating model performance...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("-" * 60)
    print(f"Training Accuracy : {train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
    print("-" * 60)
    
    # Classification Report
    print("\nClassification Report (Test Data):")
    report = classification_report(y_test, y_test_pred, target_names=class_names)
    print(report)
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print("-" * 60)
    
    # 5. Save the Model
    print("[*] Saving trained model...")
    joblib.dump(model, MODEL_PATH)
    print(f"[+] Model successfully saved to {MODEL_PATH}")
    
    # 6. Generate and save evaluation plots
    print("[*] Generating evaluation plots...")
    
    # Plot 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix - Sign Language Recognition")
    plt.tight_layout()
    cm_plot_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    
    # Plot 2: Accuracy Report (Bar Chart)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    bars = ax2.bar(['Training Accuracy', 'Testing Accuracy'], [train_acc * 100, test_acc * 100], color=['#4CAF50', '#2196F3'])
    ax2.set_ylim(0, 110)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Model Accuracy Report')
    
    # Add exact values on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    acc_plot_path = os.path.join(PLOTS_DIR, "accuracy_report.png")
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    
    print(f"[+] Confusion Matrix plot saved to {cm_plot_path}")
    print(f"[+] Accuracy Report plot saved to {acc_plot_path}")
    print("=" * 60)
    print(">>> Model Pipeline Complete! The system is ready for inference.")
    print("=" * 60)

if __name__ == "__main__":
    main()
