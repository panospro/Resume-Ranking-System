import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# === Evaluation ===
def evaluate_classifier(y_true, y_pred):
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"🎯 Accuracy: {accuracy_score(y_true, y_pred):.4f}")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
    plt.title("Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# === Training ===
def train_model(df: pd.DataFrame, save_path="pipeline.pkl"):
    print("📈 Training XGBoost classifier with SMOTE...")

    # === Feature prep
    drop_cols = ["JD_ID", "Resume_ID", "Label", "tech_stack_overlap", "soft_stack_overlap", "soft_matching_skill_count"]
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df["Label"].astype(int).replace({2: 2, 3: 2, 4: 2, 5: 2})  # Merge 3 and 4 into class 3

    # === Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"📦 Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X.shape[1]}")

    # === Define pipeline
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=0,
        reg_lambda=1.0,
        min_child_weight=2,
        eval_metric="mlogloss",
        use_label_encoder=False,
        verbosity=0,
        random_state=42
    )

    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(sampling_strategy="not majority", random_state=42)),
        ("classifier", model)
    ])

    # === Fit with early stopping using a manual call
    X_train_scaled = StandardScaler().fit_transform(X_train)
    smote = SMOTE(sampling_strategy="not majority", random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Split again for early stopping eval
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.1, stratify=y_resampled, random_state=42
    )

    model.fit(X_train_final, y_train_final)

    # === Evaluate
    y_pred = model.predict(StandardScaler().fit(X_train).transform(X_test))
    evaluate_classifier(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    # plot_learning_curve(pipeline, X, y)

    # === Save pipeline
    full_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(sampling_strategy="not majority", random_state=42)),
        ("classifier", model)
    ])
    full_pipeline.fit(X_train, y_train)
    joblib.dump(full_pipeline, save_path)
    print(f"💾 Pipeline saved to {save_path}")
    return full_pipeline
