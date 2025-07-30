import pandas as pd
import numpy as np
import joblib
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, confusion_matrix, make_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['id', 'Name', 'City', 'Profession'])
    df['Depression'].fillna(df['Depression'].mode()[0], inplace=True)
    return df

# Preprocessing
def preprocess(df):
    X = df.drop('Depression', axis=1)
    y = df['Depression']

    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Feature scaling
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y

# Train models
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'LogisticRegression': (LogisticRegression(max_iter=1000), {'C': [0.1, 1, 10]}),
        'DecisionTree': (DecisionTreeClassifier(), {'max_depth': [3, 5, 10]}),
        'RandomForest': (RandomForestClassifier(), {'n_estimators': [50, 100], 'max_depth': [5, 10]}),
        'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {'n_estimators': [50, 100], 'max_depth': [3, 6]}),
        'LinearSVC': (LinearSVC(max_iter=10000), {'C': [0.1, 1]}),
        'KNN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]})
    }

    best_model = None
    best_score = 0
    f1_scores = {}
    training_times = {}

    for name, (model, params) in models.items():
        logging.info("Training %s...", name)
        start = time.time()
        search = RandomizedSearchCV(model, params, scoring=make_scorer(f1_score), cv=5, n_iter=5, n_jobs=-1)
        search.fit(X_train, y_train)
        elapsed = time.time() - start

        preds = search.predict(X_test)
        score = f1_score(y_test, preds)
        f1_scores[name] = score
        training_times[name] = elapsed

        logging.info("%s F1 Score: %.4f | Time: %.2fs | Best Params: %s", name, score, elapsed, search.best_params_)

        # Save best model
        if score > best_score:
            best_score = score
            best_model = search.best_estimator_

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{name}_confusion_matrix.png")
        plt.close()

    # Plot F1 scores
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()))
    plt.ylabel("F1 Score")
    plt.title("F1 Scores of Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("f1_scores.png")
    plt.close()

    # Plot training times
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(training_times.keys()), y=list(training_times.values()))
    plt.ylabel("Training Time (s)")
    plt.title("Training Time of Models")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("training_times.png")
    plt.close()

    return best_model

# Save model
def save_model(model, path='best_model.pkl'):
    joblib.dump(model, path)
    logging.info("Model saved to %s", path)

# Main
def main():
    logging.info("Starting training pipeline with SMOTE...")
    df = load_data("train.csv")
    X, y = preprocess(df)

    # Split and apply SMOTE on training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    logging.info("SMOTE applied. Training samples: %d", len(y_train))

    best_model = train_models(X_train, y_train, X_test, y_test)
    save_model(best_model)
    logging.info("Pipeline completed.")

if __name__ == "__main__":
    main()
