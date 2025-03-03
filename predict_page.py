import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def show_predict_page():
    # Load the model and encoders when the page is rendered
    data = load_model()
    regressor = data["model"]  # Adjust to "regressor" if that's the key in your pickle file
    le_country = data["le_country"]
    le_education = data["le_education"]

    st.title("Software Developer Salary Prediction")
    st.write("### We need some information to predict the salary")

    countries = (
        "United States of America",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)  # Fixed typo

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience]])  # Fixed typo
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")

# If this is the main file, you might want to call show_predict_page() here
# Otherwise, it’s imported and called from app.py
if __name__ == "__main__":
    show_predict_page()