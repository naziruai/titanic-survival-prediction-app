
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction App")
st.markdown("Welcome! This app will predict whether a passenger would survive the Titanic disaster based on the information you provide.")

# Load model and scaler
@st.cache_resource
def load_model():
    return joblib.load('titanic_model.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')

model = load_model()
scaler = load_scaler()

# Place input widgets in sidebar
st.sidebar.header("🔢 Passenger Information Input")

pclass = st.sidebar.selectbox("Class (Pclass)", [1, 2, 3], 
                               format_func=lambda x: "First Class" if x == 1 else "Second Class" if x == 2 else "Third Class")
sex = st.sidebar.selectbox("Gender (Sex)", ["Female", "Male"])
age = st.sidebar.slider("Age (Years)", 0, 100, 30)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.sidebar.number_input("Parents/Children Aboard (Parch)", 0, 6, 0)
fare = st.sidebar.number_input("Ticket Fare (Fare)", 0.0, 600.0, 32.0)

# Display entered information
st.write("### Information You Entered:")
col1, col2 = st.columns(2)
with col1:
    st.write(f"- **Class:** {pclass}")
    st.write(f"- **Gender:** {sex}")
    st.write(f"- **Age:** {age}")
with col2:
    st.write(f"- **Siblings/Spouses:** {sibsp}")
    st.write(f"- **Parents/Children:** {parch}")
    st.write(f"- **Ticket Fare:** £{fare:.2f}")

# Prediction button
if st.button("🔮 Predict Survival", type="primary"):
    # Convert gender to numeric
    sex_numeric = 1 if sex == "Female" else 0
    
    # Prepare data for model
    input_data = np.array([[pclass, sex_numeric, age, sibsp, parch, fare]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Result:")
    
    if prediction == 1:
        st.success(f"### ✅ Passenger WILL SURVIVE!")
        st.metric("Survival Probability", f"{prediction_proba[1]:.1%}")
        
        # Add progress bar to show probability
        st.progress(prediction_proba[1])
        st.caption(f"Death Probability: {prediction_proba[0]:.1%}")
    else:
        st.error(f"### ❌ Passenger WILL NOT SURVIVE!")
        st.metric("Death Probability", f"{prediction_proba[0]:.1%}")
        
        # Add progress bar to show probability
        st.progress(prediction_proba[0])
        st.caption(f"Survival Probability: {prediction_proba[1]:.1%}")
    
    # Add explanation about factors affecting survival
    st.markdown("---")
    st.info("💡 **What influenced this prediction:** " + 
            ("Women are more likely to survive" if sex == "Female" else "Men have lower survival rates") + 
            f", and passengers in class {pclass} " + 
            ("have a good chance" if pclass == 1 else "have a lower chance"))

# Add footer information
st.markdown("---")
st.caption("App built with Streamlit and Logistic Regression model on the Titanic dataset")
st.image('images/confusion_matrix.png', caption='Model Confusion Matrix')
