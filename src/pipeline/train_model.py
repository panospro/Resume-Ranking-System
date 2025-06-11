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

def compare_models(X_train, X_test, y_train, y_test, feature_names=None):
    models = {
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(verbosity=0)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "R²": r2,
            "MSE": mse,
            "MAE": mae
        })

        # Optional: Plot predicted vs actual
        plt.figure(figsize=(5, 4))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{name} - R²: {r2:.3f}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    df_results = pd.DataFrame(results).sort_values("R²", ascending=False)
    print("\n📊 Model Comparison:")
    print(df_results.to_string(index=False))

def print_metrics(y_test, y_pred):
    # === Metrics ===
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test + 1e-5, y_pred + 1e-5)  # Avoid div by zero
    max_err = max_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    print("\n📊 Evaluation Metrics:")
    print(f"🎯 MSE:  {mse:.4f}")
    print(f"🔍 R² Score: {r2:.4f}")
    print(f"📏 RMSE: {rmse:.4f}")
    print(f"📉 MAE:  {mae:.4f}")
    print(f"📈 MAPE: {mape:.2%}")
    print(f"🚨 Max Error: {max_err:.4f}")
    print(f"✅ Explained Variance: {evs:.4f}\n")

def plot_results(y_test, y_pred, model, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 5)
    )

    plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
    plt.legend()
    plt.title("Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("R² Score")
    plt.grid(True)
    plt.show()

    # === Visualization ===
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Label")
    plt.ylabel("Predicted Label")
    plt.title("Predicted vs Actual Labels")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Feature Importance ===
    feature_names = X.columns
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[sorted_idx], align="center")
    plt.xticks(range(len(importances)), feature_names[sorted_idx], rotation=45, ha="right")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

def train_model(df: pd.DataFrame):
    print("📈 Training model...")
    X = df.drop(columns=["JD_ID", "Resume_ID", "Label", "tech_stack_overlap", "soft_stack_overlap"])
    y = df["Label"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # === Save Model ===
    joblib.dump(model, "trained_model.pkl")
    print("💾 Model saved to trained_model.pkl")

    print_metrics(y_test, y_pred)
    plot_results(y_test, y_pred, model, X, y)
    