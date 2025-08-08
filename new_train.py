import pandas as pd
import numpy as np
import joblib
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, confusion_matrix, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ================================
# Config and logging
# ================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TARGET_COL = 'Depression'  # change to your target column if different
DROP_COLS = ['id', 'Name', 'City', 'Profession']  # dropped if present
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ================================
# Data loading
# ================================
def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = pd.read_csv(path)
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
    if df[TARGET_COL].isna().any():
        # fill target missing with mode
        df[TARGET_COL] = df[TARGET_COL].fillna(df[TARGET_COL].mode().iloc[0])
    return df

# ================================
# Preprocessor
# ================================
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a column transformer for preprocessing categorical and numerical features."""
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    # Remove target column from numerical columns if present
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)
    
    # Create transformers
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_cols),
            ('num', num_transformer, num_cols)
        ],
        remainder='drop'
    )
    
    return preprocessor

# ================================
# Model definitions
# ================================
def get_models():
    """Define the models to train."""
    models = {
        'LogisticRegression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
        'LinearSVC': LinearSVC(random_state=RANDOM_STATE, max_iter=1000),
        'KNN': KNeighborsClassifier(n_jobs=-1)
    }
    return models

# ================================
# Training functions
# ================================
def train_model(X_train, y_train, model_name, model, preprocessor):
    """Train a single model with preprocessing pipeline."""
    start_time = time.time()
    
    # Create pipeline with SMOTE for handling class imbalance
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logging.info(f"{model_name} training completed in {training_time:.2f} seconds")
    
    return pipeline, training_time

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    logging.info(f"{model_name} - F1 Score: {f1:.4f}")
    
    return f1, cm

# ================================
# Visualization functions
# ================================
def plot_confusion_matrix(cm, model_name, save_path='visualizations/'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_path}{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_f1_scores(f1_scores, save_path='visualizations/'):
    """Plot F1 scores comparison."""
    plt.figure(figsize=(10, 6))
    models = list(f1_scores.keys())
    scores = list(f1_scores.values())
    
    bars = plt.bar(models, scores, color='skyblue', edgecolor='navy')
    plt.title('F1 Scores Comparison')
    plt.xlabel('Models')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}f1_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_times(training_times, save_path='visualizations/'):
    """Plot training times comparison."""
    plt.figure(figsize=(10, 6))
    models = list(training_times.keys())
    times = list(training_times.values())
    
    bars = plt.bar(models, times, color='lightcoral', edgecolor='darkred')
    plt.title('Training Times Comparison')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}training_times.png', dpi=300, bbox_inches='tight')
    plt.close()

# ================================
# Main training function
# ================================
def main():
    """Main training function."""
    logging.info("Starting model training process...")
    
    # Load data
    try:
        df = load_data('train.csv')
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error("train.csv not found in current directory")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    
    # Prepare features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Build preprocessor
    preprocessor = build_preprocessor(X_train)
    
    # Get models
    models = get_models()
    
    # Training results storage
    f1_scores = {}
    training_times = {}
    best_model = None
    best_f1 = 0
    
    # Train and evaluate each model
    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")
        
        try:
            trained_model, training_time = train_model(X_train, y_train, model_name, model, preprocessor)
            f1, cm = evaluate_model(trained_model, X_test, y_test, model_name)
            
            f1_scores[model_name] = f1
            training_times[model_name] = training_time
            
            # Plot confusion matrix
            plot_confusion_matrix(cm, model_name)
            
            # Track best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = trained_model
            
        except Exception as e:
            logging.error(f"Error training {model_name}: {e}")
            continue
    
    # Save best model
    if best_model is not None:
        joblib.dump(best_model, 'alternate_model.pkl')
        logging.info(f"Best model saved as 'alternate_model.pkl' with F1 score: {best_f1:.4f}")
    
    # Create visualizations
    if f1_scores:
        plot_f1_scores(f1_scores)
        plot_training_times(training_times)
        logging.info("Visualizations saved in 'visualizations/' directory")
    
    # Print summary
    logging.info("Training completed!")
    logging.info("F1 Scores:")
    for model, score in f1_scores.items():
        logging.info(f"  {model}: {score:.4f}")

if __name__ == "__main__":
    main()