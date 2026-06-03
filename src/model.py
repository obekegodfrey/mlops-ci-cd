# src/model.py
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

def train_and_save_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # 👈 Fixed: Saves the model inside the src/ folder cleanly
    model_path = BASE_DIR / "iris_model.pkl"
    joblib.dump(model, model_path)
    print(f" Model trained and saved as {model_path}")

if __name__ == "__main__":
    train_and_save_model()