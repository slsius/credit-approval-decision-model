import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from scipy.stats import uniform


DATA_PATH = "cc_approvals.data"
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df.replace("?", np.nan, inplace=True)
    df.columns = [f"A{i}" for i in range(df.shape[1])]
    return df


def main():
    df = load_data(DATA_PATH)

    # Target is last column (A15)
    target_col = df.columns[-1]
    y = df[target_col].map({"+": 1, "-": 0})

    X = df.drop(columns=[target_col]).copy()

    # Explicit numeric columns based on dataset structure
    numeric_features = ["A1", "A2", "A7", "A10", "A12"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # Convert numeric columns safely
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Logistic Regression baseline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=3000, solver="lbfgs"))
    ])

    param_dist = {"model__C": uniform(loc=0.01, scale=4.0)}

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=25,
        cv=5,
        scoring="roc_auc",
        random_state=RANDOM_STATE
    )

    search.fit(X_train, y_train)

    print("Best Parameters:", search.best_params_)
    print("Best CV ROC_AUC:", round(search.best_score_, 3))

    y_pred = search.predict(X_test)
    y_prob = search.predict_proba(X_test)[:, 1]

    print("Test ROC_AUC:", round(roc_auc_score(y_test, y_prob), 3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
