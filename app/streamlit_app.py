"""
Streamlit dashboard: risk scoring UI + what-if analysis + business insights.

Run with: streamlit run app/streamlit_app.py
"""
import sys
import os
import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from config import MODEL_PATH, PREPROCESSOR_PATH, MEDICATION_COLS  # noqa: E402
from preprocessing import preprocess  # noqa: E402
from feature_engineering import add_engineered_features  # noqa: E402

st.set_page_config(page_title="ReadmitGuard", layout="wide")
st.title("🏥 ReadmitGuard — 30-Day Readmission Risk Dashboard")
st.caption("Explainable risk scoring for diabetic inpatients, built on the UCI "
           "Diabetes 130-US Hospitals dataset.")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preproc = joblib.load(PREPROCESSOR_PATH)
    return model, preproc


def score_patient(patient_dict, model, preproc):
    df = pd.DataFrame([patient_dict])
    for col in MEDICATION_COLS:
        if col not in df.columns:
            df[col] = "No"
    df = add_engineered_features(df)
    X, _, _, _ = preprocess(df, fit=False, scaler=preproc["scaler"],
                             encoders=preproc["encoders"])
    return float(model.predict_proba(X)[0, 1])


tab1, tab2 = st.tabs(["🔍 Patient Risk Scoring (What-If Analysis)", "📊 Business Insights"])

with tab1:
    st.subheader("Adjust patient details to see risk change in real time")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.selectbox("Age group", [
            "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
            "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
        ], index=7)
        gender = st.selectbox("Gender", ["Female", "Male"])
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic",
                                      "Asian", "Other"])
        time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 4)

    with col2:
        num_medications = st.slider("Number of medications", 1, 40, 15)
        num_procedures = st.slider("Number of procedures", 0, 6, 1)
        num_lab_procedures = st.slider("Number of lab procedures", 1, 100, 45)
        number_diagnoses = st.slider("Number of diagnoses", 1, 16, 7)

    with col3:
        number_inpatient = st.slider("Prior inpatient visits", 0, 10, 0)
        number_emergency = st.slider("Prior emergency visits", 0, 10, 0)
        number_outpatient = st.slider("Prior outpatient visits", 0, 10, 0)
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        diabetesMed = st.selectbox("On diabetes medication?", ["Yes", "No"])

    patient = dict(
        race=race, gender=gender, age=age,
        admission_type_id=1, discharge_disposition_id=1, admission_source_id=7,
        time_in_hospital=time_in_hospital, num_lab_procedures=num_lab_procedures,
        num_procedures=num_procedures, num_medications=num_medications,
        number_outpatient=number_outpatient, number_emergency=number_emergency,
        number_inpatient=number_inpatient, number_diagnoses=number_diagnoses,
        diag_1="428", diag_2="250", diag_3="401",
        max_glu_serum="None", A1Cresult="None", change="No",
        diabetesMed=diabetesMed, insulin=insulin,
    )

    if st.button("Score Patient", type="primary"):
        try:
            model, preproc = load_artifacts()
            proba = score_patient(patient, model, preproc)
            risk = "🔴 High" if proba >= 0.5 else ("🟡 Medium" if proba >= 0.25 else "🟢 Low")
            m1, m2 = st.columns(2)
            m1.metric("30-Day Readmission Probability", f"{proba:.1%}")
            m2.metric("Risk Level", risk)
            if proba >= 0.5:
                st.warning("Recommend: schedule a follow-up call within 7 days of "
                           "discharge and review medication plan.")
        except FileNotFoundError:
            st.error("Model not found. Run `python src/run_pipeline.py` first to "
                     "train and save the model.")

with tab2:
    st.subheader("Business Insights")
    st.markdown("""
    - **~11%** of encounters in the training data resulted in a readmission within
      30 days — the primary cost driver this model targets.
    - Patients with **more prior inpatient/emergency visits** and **more medication
      changes during their stay** show the highest predicted risk — these are the
      strongest candidates for proactive discharge planning.
    - Recommended action: route patients scored **High risk** to a nurse-led
      follow-up call within 48–72 hours of discharge; **Medium risk** patients get
      an automated check-in text/call.
    - Re-run `src/eda.py` after retraining to regenerate the figures used in the
      report (`reports/figures/`).
    """)
    st.info("Swap this section for your team's actual EDA figures/insights once "
            "the pipeline has been run on the real dataset.")
