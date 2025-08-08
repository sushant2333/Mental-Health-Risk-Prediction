import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Mental Health Risk Prediction",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 Mental Health Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    """Load the trained model with error handling."""
    try:
        model = joblib.load("alternate_model.pkl")
        return model, None
    except FileNotFoundError:
        return None, "Model file 'alternate_model.pkl' not found. Please run the training script first."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# Load the model
model, error = load_model()

if error:
    st.error(error)
    st.info("Please run the training script (new_train.py) to generate the model file.")
    st.stop()

# Input Parameters Section
st.header("📋 Input Parameters")
st.markdown("Adjust the values below to get your prediction:")

# Create columns for better layout
col1_input, col2_input, col3_input = st.columns(3)

with col1_input:
    st.subheader("👤 Personal Information")
    age = st.slider("Age", 10, 100, 25, help="Select your age")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Select your gender")
    wp_or_student = st.selectbox("Status", ["Student", "Working Professional"], help="Are you a student or working professional?")
    
    st.subheader("🎓 Academic/Work Information")
    academic_pressure = st.slider("Academic Pressure", 1, 5, 1, 
                                 help="Rate your academic pressure (1=Low, 5=High)")
    work_pressure = st.slider("Work Pressure", 1, 5, 1,
                             help="Rate your work pressure (1=Low, 5=High)")
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.0, step=0.1,
                     help="Your current CGPA or GPA")

with col2_input:
    st.subheader("🌙 Lifestyle Information")
    sleep_duration = st.slider("Sleep Duration (hours)", 0, 12, 6,
                              help="Average hours of sleep per day")
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy"],
                                 help="Rate your dietary habits")
    work_study_hours = st.slider("Work/Study Hours", 0, 20, 8,
                                help="Average hours spent working or studying per day")
    
    st.subheader("🧠 Mental Health Factors")
    suicidal_thoughts = st.selectbox("Suicidal Thoughts", ["No", "Yes"],
                                   help="Have you ever had suicidal thoughts?")
    financial_stress = st.slider("Financial Stress", 1, 5, 1,
                               help="Rate your financial stress (1=Low, 5=High)")
    family_history = st.selectbox("Family History", ["No", "Yes"],
                                help="Family history of mental illness?")

with col3_input:
    st.subheader("📊 Satisfaction Levels")
    study_satisfaction = st.slider("Study Satisfaction", 1, 5, 1,
                                  help="Rate your satisfaction with studies (1=Low, 5=High)")
    job_satisfaction = st.slider("Job Satisfaction", 1, 5, 1,
                                help="Rate your job satisfaction (1=Low, 5=High)")
    degree = st.selectbox("Degree", ["Bachelor's", "Master's", "PhD", "Other"],
                         help="Your current or highest degree")

# Prediction button - centered below input parameters
st.markdown("---")
col_pred, col_pred2, col_pred3 = st.columns([1, 2, 1])

with col_pred2:
    if st.button("🔮 Get Prediction", type="primary", use_container_width=True):
        st.session_state.prediction_clicked = True

st.markdown("---")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Prediction Results")
    
    # Prepare input data
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
    
    # Prediction results
    if st.session_state.get('prediction_clicked', False):
        with st.spinner("Analyzing your data..."):
            try:
                # Make prediction
                prediction = model.predict(input_data)[0]
                prediction_proba = model.predict_proba(input_data)[0]
                
                # Display result
                if prediction == 1:
                    risk_level = "High Risk"
                    risk_class = "high-risk"
                    risk_color = "#f44336"
                    confidence = prediction_proba[1]
                else:
                    risk_level = "Low Risk"
                    risk_class = "low-risk"
                    risk_color = "#4caf50"
                    confidence = prediction_proba[0]
                
                # Display prediction box
                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h2 style="color: {risk_color}; margin: 0;">{risk_level} of Depression</h2>
                    <p style="margin: 0.5rem 0;">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk factors analysis
                st.subheader("🔍 Risk Factor Analysis")
                
                # Calculate risk factors
                risk_factors = []
                if academic_pressure >= 4:
                    risk_factors.append("High academic pressure")
                if work_pressure >= 4:
                    risk_factors.append("High work pressure")
                if financial_stress >= 4:
                    risk_factors.append("High financial stress")
                if sleep_duration < 6:
                    risk_factors.append("Insufficient sleep")
                if work_study_hours > 12:
                    risk_factors.append("Excessive work/study hours")
                if suicidal_thoughts == "Yes":
                    risk_factors.append("History of suicidal thoughts")
                if family_history == "Yes":
                    risk_factors.append("Family history of mental illness")
                if dietary_habits == "Unhealthy":
                    risk_factors.append("Unhealthy dietary habits")
                
                if risk_factors:
                    st.warning("⚠️ Identified Risk Factors:")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("✅ No significant risk factors identified")
                
                # Generate report
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                report_data = {
                    "Timestamp": timestamp,
                    "Prediction": risk_level,
                    "Confidence": f"{confidence:.1%}",
                    **input_data.iloc[0].to_dict()
                }
                
                # Save report
                report_df = pd.DataFrame([report_data])
                report_df.to_csv("prediction_report.csv", mode='a', header=not os.path.exists("prediction_report.csv"), index=False)
                
                st.success("📄 Report saved to prediction_report.csv")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please ensure the model is compatible with the input data format.")
        
        # Reset the session state
        st.session_state.prediction_clicked = False

with col2:
    st.header("📈 Quick Insights")
    
        # Metrics cards
    col_a, col_b = st.columns(2)
    
    with col_a:
        stress_level = max(academic_pressure, work_pressure, financial_stress)
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #333; margin: 0;">Stress Level</h4>
            <h2 style="color: #1f77b4; margin: 0.5rem 0;">{stress_level}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        sleep_quality = "Good" if sleep_duration >= 7 else "Poor"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #333; margin: 0;">Sleep Quality</h4>
            <h2 style="color: #1f77b4; margin: 0.5rem 0;">{sleep_quality}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Time distribution pie chart
    st.subheader("⏰ Daily Time Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['Work/Study', 'Sleep', 'Other']
    sizes = [work_study_hours, sleep_duration, max(0, 24 - work_study_hours - sleep_duration)]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    if sum(sizes) > 0:
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.info("Adjust time values to see distribution")
    
    # Stress factors bar chart
    st.subheader("📊 Stress Factors")
    stress_data = {
        'Academic': academic_pressure,
        'Work': work_pressure,
        'Financial': financial_stress
    }
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(stress_data.keys(), stress_data.values(), color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax.set_ylim(0, 5)
    ax.set_ylabel('Level (1-5)')
    ax.set_title('Stress Levels')
    
    # Add value labels on bars
    for bar, value in zip(bars, stress_data.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(value), ha='center', va='bottom')
    
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>⚠️ This tool is for educational purposes only and should not replace professional medical advice.</p>
    <p>If you're experiencing mental health concerns, please seek help from a qualified healthcare provider.</p>
</div>
""", unsafe_allow_html=True)

# Additional features
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    **Model Information:**
    - **Algorithm**: Ensemble of multiple machine learning models
    - **Training Data**: Mental health survey responses
    - **Features**: 15 different factors including lifestyle, academic, and mental health indicators
    - **Purpose**: Educational tool for understanding mental health risk factors
    
    **How to Use:**
    1. Adjust the input parameters in the sidebar
    2. Click "Get Prediction" to analyze your risk level
    3. Review the risk factor analysis and insights
    4. Consider the recommendations provided
    
    **Disclaimer**: This is a research tool and should not be used for clinical diagnosis.
    """)

with st.expander("📊 View Sample Data"):
    try:
        sample_data = pd.read_csv("train.csv")
        st.dataframe(sample_data.head(10))
        st.info(f"Dataset contains {len(sample_data)} records with {len(sample_data.columns)} features")
    except FileNotFoundError:
        st.warning("Sample data file not found")
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
