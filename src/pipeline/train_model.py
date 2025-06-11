import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, max_error, explained_variance_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor

# === Metrics ===
def evaluate_model(y_true, y_pred, verbose=True):
    metrics = {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        # "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        # "MAE": mean_absolute_error(y_true, y_pred),
        # "MAPE": mean_absolute_percentage_error(y_true + 1e-5, y_pred + 1e-5),
        # "MaxError": max_error(y_true, y_pred),
        # "ExplainedVar": explained_variance_score(y_true, y_pred)
    }
    if verbose:
        print("\n📊 Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    return metrics

# === Visualizations ===
def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[sorted_idx], align="center")
        plt.xticks(range(len(importances)), np.array(feature_names)[sorted_idx], rotation=45, ha="right")
        plt.title("Feature Importances")
        plt.tight_layout()
        plt.show()

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
    plt.title("Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("R² Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Multi-model comparison ===
def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=0.01),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(verbosity=0)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, verbose=False)
        results.append({"Model": name, **metrics})

    df_results = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\n📊 Model Comparison:")
    print(df_results.to_string(index=False))

# === Model training ===
def train_model(df: pd.DataFrame, save_path="trained_model.pkl"):
    print("📈 Training model with XGBoost...")
    
    # Feature setup
    X = df.drop(columns=["JD_ID", "Resume_ID", "Label", "tech_stack_overlap", "soft_stack_overlap"])
    y = df["Label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and fit best model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbosity=0
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Save model
    joblib.dump(model, save_path)
    print(f"💾 Model saved to {save_path}")

    # Report + plots
    evaluate_model(y_test, y_pred)
    plot_predictions(y_test, y_pred)
    plot_feature_importances(model, X.columns)
    plot_learning_curve(model, X, y)

    return model
