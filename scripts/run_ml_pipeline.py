import os
import pandas as pd
import numpy as np
import joblib # For saving/loading the ML model

# Import modules from src/
from src.ml_pipeline import (
    load_and_prepare_features,
    split_data,
    select_top_features,
    scale_features,
    train_and_evaluate_ml_model,
    plot_feature_selection_vs_accuracy,
    run_kfold_cross_validation,
    run_shap_analysis,
    get_voting_classifier # Needed if you want to load the model later for SHAP
)

def main():
    # --- Configuration ---
    FEATURES_CSV_PATH = 'data/trained_mobilenet_cbam_features.csv' # Path to your extracted features CSV
    ML_MODEL_SAVE_PATH = 'trained_models/voting_classifier_best.pkl'
    TOP_K_FEATURES = 240 # The optimal number of features found
    SHAP_NUM_SAMPLES = 25 # Number of samples for SHAP analysis (can be computationally intensive)
    
    # Ensure save directory for ML model exists
    os.makedirs(os.path.dirname(ML_MODEL_SAVE_PATH), exist_ok=True)

    if not os.path.exists(FEATURES_CSV_PATH):
        print(f"Error: Feature CSV not found at {FEATURES_CSV_PATH}.")
        print("Please run 'python scripts/run_inference.py' first to extract features.")
        return

    # --- Load and Prepare Data ---
    df_shuffled, label_encoder = load_and_prepare_features(FEATURES_CSV_PATH)
    
    # --- Split Data ---
    X_train, y_train, X_test, y_test = split_data(df_shuffled)

    # --- Feature Selection (using Renyi Entropy) ---
    # Need to compute entropies on the full dataset first to get sorted_features
    # This step should ideally be part of a preprocessing stage or done once and stored
    print("Pre-calculating feature entropies on full dataset for consistent sorting...")
    feature_entropies = {feature: renyi_entropy(df_shuffled[feature].values, alpha=2) for feature in df_shuffled.drop(columns=['label']).columns}
    sorted_features = sorted(feature_entropies, key=feature_entropies.get, reverse=True)
    
    top_features = select_top_features(X_train, TOP_K_FEATURES) # Use X_train for entropy calculation as per your original code
    # Ensure top_features are derived from the *original* column names, which they are here.

    # --- Scale Selected Features ---
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, top_features)

    # --- Train and Evaluate ML Model ---
    print("\n--- Training and Evaluating Main ML Model ---")
    trained_voting_clf = train_and_evaluate_ml_model(
        X_train_scaled, y_train, X_test_scaled, y_test, label_encoder, 
        model_save_path=ML_MODEL_SAVE_PATH
    )

    # --- Plot Feature Selection vs. Accuracy ---
    print("\n--- Running Feature Selection vs. Accuracy Analysis ---")
    feature_counts_to_test = [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    plot_feature_selection_vs_accuracy(df_shuffled, label_encoder, sorted_features, feature_counts_to_test)
    
    # --- K-Fold Cross-Validation ---
    print("\n--- Running K-Fold Cross-Validation ---")
    run_kfold_cross_validation(df_shuffled, top_features, label_encoder, n_splits=5)

    # --- SHAP Analysis ---
    print("\n--- Performing SHAP Analysis ---")
    # Reload the best model if it was saved and you need a fresh instance,
    # or just use 'trained_voting_clf' if it's the one that was just trained and saved.
    # For robust SHAP analysis, it's good practice to use the model that will be deployed,
    # which is the one saved to ML_MODEL_SAVE_PATH.
    
    # If the voting_clf from train_and_evaluate_ml_model is not the one you want for SHAP, load it:
    # loaded_voting_clf = joblib.load(ML_MODEL_SAVE_PATH)
    # run_shap_analysis(loaded_voting_clf, X_train_scaled, X_test_scaled, top_features, label_encoder, num_samples=SHAP_NUM_SAMPLES)
    
    # Otherwise, just use the already trained object:
    run_shap_analysis(trained_voting_clf, X_train_scaled, X_test_scaled, top_features, label_encoder, num_samples=SHAP_NUM_SAMPLES)

    print("\nML pipeline execution complete!")

if __name__ == "__main__":
    main()