import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model(df: pd.DataFrame):
    print("📈 Training model...")
    X = df.drop(columns=["JD_ID", "Resume_ID", "Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"🎯 MSE on test set: {mse:.3f}")
    print(f"📊 R^2 Score: {r2:.3f}")

    # === Save Model ===
    joblib.dump(model, "trained_model.pkl")
    print("💾 Model saved to trained_model.pkl")

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
