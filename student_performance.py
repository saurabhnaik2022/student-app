import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open("student_lr_final_model.pkl", 'rb') as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le

def preprocessing_data(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data=preprocessing_data(data,scaler,le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student performance Prediction")
    st.write("enter your data to get student performance")

    hours_edited = st.number_input("Hours studied", min_value=1, value=5)
    previous_score=st.number_input("previous score", max_value=100, value=45)
    extra=st.selectbox("Extra curry activity",['Yes','No'])
    sleep_hr=st.number_input("Sleeping hours", min_value=4,max_value=10, value=7)
    papers_solved=st.number_input("no of question solved", min_value=1,max_value=10, value=5)

    if st.button('predict_your_score'):
        user_data={
            'Hours Studied':hours_edited, 
            'Previous Scores':previous_score, 
            'Extracurricular Activities':extra,
            'Sleep Hours':sleep_hr,
            'Sample Question Papers Practiced':papers_solved
        }
        prediction = predict_data(user_data)
        st.success(f"Your prediction result is {prediction}")

if __name__ == "__main__":
    main()

