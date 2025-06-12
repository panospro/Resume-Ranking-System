import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def evaluate_classifier(y_true, y_pred):
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")


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


def train_model(df: pd.DataFrame, save_path="final_xgb_model.pkl"):
    print("📈 Training final XGBoost classifier...")

    # === Prepare features
    X = df.drop(columns=["JD_ID", "Resume_ID", "Label", "tech_stack_overlap", "soft_stack_overlap", "soft_matching_skill_count"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    y = df["Label"].astype(int).replace({
        0: 0,  # 0 stays 0
        1: 0,  # merged into 0
        2: 1,  # shifted to 1
        3: 2,  # merged into 2
        4: 2   # merged into 2
    })

    # === Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"📦 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]} | Features: {X_train.shape[1]}")

    # === Apply targeted SMOTE to minority classes
    smote = SMOTE(sampling_strategy="not majority", random_state=42)

    # === Define model using best params + regularization
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
        ("smote", smote),
        ("classifier", model)
    ])

    # Reuse the scaler from training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Resample and fit
    smote_X, smote_y = smote.fit_resample(X_train_scaled, y_train)

    model.fit(smote_X, smote_y)

    # === Predict and evaluate
    y_pred = model.predict(StandardScaler().fit_transform(X_test))
    evaluate_classifier(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    plot_learning_curve(model, StandardScaler().fit_transform(X), y)

    # === Save
    joblib.dump(model, save_path)
    print(f"💾 Model saved to {save_path}")
    return model