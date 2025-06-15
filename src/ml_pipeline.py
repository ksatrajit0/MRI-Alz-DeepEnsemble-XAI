import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import joblib # For saving/loading scikit-learn models
import shap # For SHAP explanations

def load_and_prepare_features(csv_path, random_state=42):
    """
    Loads features from CSV, encodes labels, and shuffles the dataset.

    Args:
        csv_path (str): Path to the CSV file containing features and labels.
        random_state (int): Seed for shuffling.

    Returns:
        tuple: (df_shuffled, label_encoder)
    """
    df = pd.read_csv(csv_path)
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    df_shuffled = shuffle(df, random_state=random_state)
    print("Features loaded and prepared.")
    return df_shuffled, label_encoder

def split_data(df, test_frac=0.2, random_state=42):
    """
    Splits the shuffled DataFrame into training and testing sets,
    maintaining class distribution for training.

    Args:
        df (pd.DataFrame): Shuffled DataFrame with features and 'label' column.
        test_frac (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for sampling.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    train_df = df.groupby('label', group_keys=False, as_index=False).apply(
        lambda x: x.sample(frac=(1 - test_frac), random_state=random_state)
    )
    test_df = df.drop(train_df.index)

    X_train, y_train = train_df.drop(columns=['label']), train_df['label']
    X_test, y_test = test_df.drop(columns=['label']), test_df['label']
    print(f"Data split into train ({len(X_train)} samples) and test ({len(X_test)} samples).")
    return X_train, y_train, X_test, y_test

def renyi_entropy(feature_values, alpha=2):
    """Compute Rényi entropy for a single feature."""
    probabilities, counts = np.unique(feature_values, return_counts=True)
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[counts > 0]
    counts = counts[counts > 0]
    
    probabilities = counts / counts.sum()

    if alpha == 1: # Shannon entropy as a limit case
        # Add a small epsilon to probabilities to avoid log(0)
        return -np.sum(probabilities * np.log(probabilities + 1e-9))
    else:
        return 1 / (1 - alpha) * np.log(np.sum(probabilities ** alpha))

def select_top_features(X_train, num_features):
    """
    Computes Rényi entropy for features and selects the top K most informative.

    Args:
        X_train (pd.DataFrame): Training features.
        num_features (int): Number of top features to select.

    Returns:
        list: Names of the selected top features.
    """
    print(f"Selecting top {num_features} features based on Rényi entropy...")
    feature_entropies = {feature: renyi_entropy(X_train[feature].values, alpha=2) for feature in X_train.columns}
    sorted_features = sorted(feature_entropies, key=feature_entropies.get, reverse=True)
    top_features = sorted_features[:num_features]
    print("Feature selection complete.")
    return top_features

def scale_features(X_train, X_test, top_features=None):
    """
    Scales features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        top_features (list, optional): List of feature names to select before scaling.
                                      If None, uses all features in X_train/X_test.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    if top_features:
        X_train_selected = X_train[top_features].copy()
        X_test_selected = X_test[top_features].copy()
    else:
        X_train_selected = X_train.copy()
        X_test_selected = X_test.copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    print("Features scaled using StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler

def get_voting_classifier():
    """Returns a pre-defined VotingClassifier."""
    return VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(class_weight='balanced', solver='saga', max_iter=3000, random_state=42)),
            ('svm', SVC(class_weight='balanced', probability=True, random_state=42)),
            ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
        ],
        voting='soft'
    )

def train_and_evaluate_ml_model(X_train_scaled, y_train, X_test_scaled, y_test, label_encoder, model_save_path=None):
    """
    Trains the VotingClassifier and evaluates its performance.

    Args:
        X_train_scaled (np.array): Scaled training features.
        y_train (pd.Series): Training labels.
        X_test_scaled (np.array): Scaled test features.
        y_test (pd.Series): Test labels.
        label_encoder (LabelEncoder): Fitted LabelEncoder for class names.
        model_save_path (str, optional): Path to save the trained model.

    Returns:
        VotingClassifier: The trained voting classifier.
    """
    voting_clf = get_voting_classifier()
    print("Training Voting Classifier...")
    voting_clf.fit(X_train_scaled, y_train)
    y_pred_voting = voting_clf.predict(X_test_scaled)
    print("Model training and inference done.")

    print("\n--- Voting Classifier Classification Report ---")
    print(classification_report(y_test, y_pred_voting, target_names=label_encoder.classes_))

    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_voting), annot=True, cmap='viridis', fmt='d',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Voting Classifier Confusion Matrix")
    plt.tight_layout()
    plt.show()

    accuracy = accuracy_score(y_test, y_pred_voting)
    precision = precision_score(y_test, y_pred_voting, average='weighted')
    recall = recall_score(y_test, y_pred_voting, average='weighted')
    f1 = f1_score(y_test, y_pred_voting, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")

    if model_save_path:
        joblib.dump(voting_clf, model_save_path)
        print(f"Trained Voting Classifier saved to {model_save_path}")

    return voting_clf

def plot_feature_selection_vs_accuracy(df_shuffled, label_encoder, sorted_features, feature_counts, random_state=42):
    """
    Plots the model accuracy as a function of the number of selected features.

    Args:
        df_shuffled (pd.DataFrame): The full shuffled DataFrame.
        label_encoder (LabelEncoder): Fitted LabelEncoder for class names.
        sorted_features (list): List of all features sorted by importance.
        feature_counts (list): List of numbers of features to test.
        random_state (int): Seed for splitting data.
    """
    print("\nEvaluating accuracy vs. number of features selected...")
    accuracies = []
    
    # Re-split data for consistent comparison across feature counts
    train_df = df_shuffled.groupby('label', group_keys=False, as_index=False).apply(lambda x: x.sample(frac=0.8, random_state=random_state))
    test_df = df_shuffled.drop(train_df.index)
    X_train_base, y_train_base = train_df.drop(columns=['label']), train_df['label']
    X_test_base, y_test_base = test_df.drop(columns=['label']), test_df['label']


    for num_features in feature_counts:
        top_features = sorted_features[:num_features]
        X_train_selected, X_test_selected, _ = scale_features(X_train_base, X_test_base, top_features)
        
        voting_clf = get_voting_classifier()
        voting_clf.fit(X_train_selected, y_train_base)
        
        accuracy = voting_clf.score(X_test_selected, y_test_base) * 100
        accuracies.append(accuracy)
        print(f"Features: {num_features}, Accuracy: {accuracy:.2f}%")

    plt.figure(figsize=(9, 6))
    plt.plot(feature_counts, accuracies, marker='o', linestyle='-', color='b', label="Model Accuracy")
    plt.xlabel("Number of Selected Features")
    plt.ylabel("Accuracy (%)")
    plt.title("Feature Selection vs. Accuracy (Voting Classifier)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_kfold_cross_validation(df_shuffled, top_features, label_encoder, n_splits=5, random_state=42):
    """
    Performs K-Fold cross-validation for the ML pipeline.

    Args:
        df_shuffled (pd.DataFrame): The full shuffled DataFrame.
        top_features (list): List of selected feature names.
        label_encoder (LabelEncoder): Fitted LabelEncoder for class names.
        n_splits (int): Number of folds for K-Fold CV.
        random_state (int): Seed for KFold.
    """
    print(f"\nRunning {n_splits}-Fold Cross-Validation...")

    X, y = df_shuffled.drop(columns=['label']), df_shuffled['label']
    X_selected = X[top_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
    best_fold = {'index': None, 'accuracy': 0, 'y_true': None, 'y_pred': None}

    voting_clf = get_voting_classifier()

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        voting_clf.fit(X_train, y_train)
        y_pred = voting_clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1-score'].append(f1)
        
        print(f"Fold {fold+1}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

        if acc > best_fold['accuracy']:
            best_fold.update({'index': fold, 'accuracy': acc, 'y_true': y_test, 'y_pred': y_pred})

    average_metrics = {key: np.mean(values) for key, values in metrics.items()}
    print("\n--- Average Cross-Validation Metrics ---")
    print(pd.DataFrame(average_metrics, index=['Average']).round(4))

    # Plot metrics across folds
    plt.figure(figsize=(14, 10))
    metrics_list = ['accuracy', 'precision', 'recall', 'f1-score']
    colors = ['b', 'g', 'r', 'm']
    for i, metric in enumerate(metrics_list):
        plt.subplot(2, 2, i+1)
        plt.plot(range(1, n_splits + 1), metrics[metric], marker='o', color=colors[i], label=metric.capitalize())
        plt.title(f"{metric.capitalize()} Curve Across Folds")
        plt.xlabel("Fold Number")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Confusion Matrix for the best fold
    if best_fold['y_true'] is not None:
        plt.figure(figsize=(7, 6))
        sns.heatmap(confusion_matrix(best_fold['y_true'], best_fold['y_pred']), annot=True, cmap='Blues', fmt='d',
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f"Confusion Matrix (Best Fold: {best_fold['index']+1})")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

        print(f"\n--- Classification Report (Best Fold: {best_fold['index']+1}) ---")
        print(classification_report(best_fold['y_true'], best_fold['y_pred'], target_names=label_encoder.classes_))
    else:
        print("No best fold found (perhaps no folds completed successfully).")

def run_shap_analysis(voting_clf, X_train_selected_scaled, X_test_selected_scaled, top_features, label_encoder, num_samples=25):
    """
    Performs SHAP analysis on the trained VotingClassifier.

    Args:
        voting_clf (VotingClassifier): The trained voting classifier.
        X_train_selected_scaled (np.array): Scaled training features (used for explainer background).
        X_test_selected_scaled (np.array): Scaled test features (for explanations).
        top_features (list): Names of the features.
        label_encoder (LabelEncoder): Fitted LabelEncoder for class names.
        num_samples (int): Number of samples from X_test to explain.
    """
    print(f"\nApplying SHAP KernelExplainer for {num_samples} samples...")
    # SHAP KernelExplainer works with model.predict_proba
    # Note: KernelExplainer can be computationally intensive for many samples or features.
    explainer = shap.KernelExplainer(voting_clf.predict_proba, X_train_selected_scaled[:num_samples])

    # Ensure X_test_selected_scaled is a DataFrame for feature_names, or pass feature_names directly
    X_test_df_for_shap = pd.DataFrame(X_test_selected_scaled[:num_samples], columns=top_features)
    
    shap_values = explainer.shap_values(X_test_df_for_shap)

    # Convert numeric labels back to class names for SHAP plots
    class_names_encoded = [label_encoder.inverse_transform([i])[0] for i in range(len(label_encoder.classes_))]

    # Summary plot for all classes
    print("Generating SHAP summary plot (all classes)...")
    shap.summary_plot(shap_values, X_test_df_for_shap, feature_names=top_features, class_names=class_names_encoded)

    # Optional: Summary plot for a specific class (e.g., class 0)
    # print("\nGenerating SHAP summary plot for class 0 (Non Demented)...")
    # shap.summary_plot(shap_values[0], X_test_df_for_shap, feature_names=top_features)

    print("SHAP analysis completed!")