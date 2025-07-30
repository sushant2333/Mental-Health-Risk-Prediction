import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Mental Health Risk Prediction", layout="centered")
st.title("🧠 Mental Health Risk Prediction App")

st.markdown("Provide input values for prediction:")

# Input fields
age = st.slider("Age", 10, 100, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
wp_or_student = st.selectbox("Working Professional or Student", ["Student", "Working Professional"])
academic_pressure = st.slider("Academic Pressure", 1, 5, 1)
work_pressure = st.slider("Work Pressure", 1, 5, 1)
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
study_satisfaction = st.slider("Study Satisfaction", 1, 5, 1)
job_satisfaction = st.slider("Job Satisfaction", 1, 5, 1)
sleep_duration = st.slider("Sleep Duration (hours)", 0, 12, 6)
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])
degree = st.selectbox("Degree", ["Bachelor's", "Master's", "PhD", "Other"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
work_study_hours = st.slider("Work/Study Hours", 0, 20, 8)
financial_stress = st.slider("Financial Stress", 1, 5, 1)
family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

# Prepare input dataframe
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Working Professional or Student': wp_or_student,
    'Academic Pressure': academic_pressure,
    'Work Pressure': work_pressure,
    'CGPA': cgpa,
    'Study Satisfaction': study_satisfaction,
    'Job Satisfaction': job_satisfaction,
    'Sleep Duration': sleep_duration,
    'Dietary Habits': dietary_habits,
    'Degree': degree,
    'Have you ever had suicidal thoughts ?': suicidal_thoughts,
    'Work/Study Hours': work_study_hours,
    'Financial Stress': financial_stress,
    'Family History of Mental Illness': family_history
}])

# Label encoding
encoded_input = input_data.copy()
label_encoders = {}
for col in encoded_input.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    encoded_input[col] = le.fit_transform(encoded_input[col])
    label_encoders[col] = le  # Store encoder if needed later

# Scaling
scaler = StandardScaler()
input_scaled = scaler.fit_transform(encoded_input)

# Predict
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    result = "High Risk of Depression" if pred == 1 else "Low Risk of Depression"
    st.subheader(f"🧾 Prediction: {result}")

    # Save to HTML
    report_html = f"""
    <html>
    <head><title>Prediction Report</title></head>
    <body>
        <h2>Prediction Report</h2>
        <p><strong>Prediction:</strong> {result}</p>
        <h4>Input Features:</h4>
        {input_data.to_html(index=False)}
    </body>
    </html>
    """
    report_path = "prediction_report.html"
    with open(report_path, "w") as f:
        f.write(report_html)
    st.success("Report saved as prediction_report.html")

    # 📊 Custom Visualizations
    st.markdown("### 📊 Visual Summary Based on Your Input")

    # Bar chart: stress & satisfaction levels
    stress_factors = {
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "Financial Stress": financial_stress,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction
    }

    fig_stress, ax_stress = plt.subplots()
    ax_stress.bar(stress_factors.keys(), stress_factors.values(), color='teal')
    ax_stress.set_ylabel("Level (1 to 5)")
    ax_stress.set_ylim(0, 5)
    ax_stress.set_title("Stress & Satisfaction Levels")
    plt.xticks(rotation=30)
    st.pyplot(fig_stress)

    # Pie chart: Time distribution
    fig_pie, ax_pie = plt.subplots()
    pie_labels = ['Work/Study Hours', 'Sleep Duration', 'Other Activities']
    other_hours = 24 - work_study_hours - sleep_duration
    ax_pie.pie([work_study_hours, sleep_duration, max(0, other_hours)],
               labels=pie_labels,
               autopct='%1.1f%%',
               colors=['skyblue', 'lightgreen', 'lightcoral'],
               startangle=90)
    ax_pie.set_title("Daily Time Distribution")
    st.pyplot(fig_pie)

    # 🧠 SHAP Explainability
    st.markdown("### 🧠 SHAP Explanation of Prediction")
    try:
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(encoded_input)

        # SHAP force plot (can be shown as a matplotlib plot for Streamlit)
        # st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.initjs()
        fig_shap, ax_shap = plt.subplots(figsize=(10, 4))
        shap.summary_plot(shap_values, encoded_input, plot_type="bar", show=False)
        st.pyplot(fig_shap)
    except Exception as e:
        st.warning("SHAP explanation could not be generated.")
        st.error(str(e))

# Sample visualizations from dataset
if st.checkbox("Show Sample Visualizations"):
    sample_df = pd.read_csv("train.csv")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=sample_df, x='Depression', ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.histplot(data=sample_df, x='Age', hue='Depression', kde=True, ax=ax2)
    st.pyplot(fig2)
