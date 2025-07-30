import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_test_data(path):
    df = pd.read_csv(path)
    df_ids = df[['id']]  # Save ID for output
    df = df.drop(columns=['id', 'Name', 'City', 'Profession'], errors='ignore')
    return df, df_ids

def preprocess(df):
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df

def plot_prediction_distribution(preds):
    # Bar Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x=preds)
    plt.title("Prediction Count (0: No Depression, 1: Depression)")
    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.xticks([0, 1])
    plt.tight_layout()
    plt.savefig("prediction_bar_plot.png")
    plt.close()

    # Pie Chart
    class_labels = ['No Depression', 'Depression']
    counts = pd.Series(preds).value_counts().sort_index()
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=class_labels, autopct='%1.1f%%', startangle=90, colors=["skyblue", "salmon"])
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig("prediction_pie_chart.png")
    plt.close()

def main():
    logging.info("🔍 Loading model and test data for prediction...")
    model = joblib.load("best_model.pkl")
    df, df_ids = load_test_data("test.csv")
    X_test = preprocess(df)

    preds = model.predict(X_test)

    # Save prediction results
    output = pd.DataFrame({
        'id': df_ids['id'],
        'Predicted_Depression': preds
    })
    output.to_csv("test_predictions.csv", index=False)
    logging.info("✅ Predictions saved to test_predictions.csv")

    # Plot graphs
    plot_prediction_distribution(preds)
    logging.info("📊 Prediction graphs saved (bar + pie).")

if __name__ == "__main__":
    main()
