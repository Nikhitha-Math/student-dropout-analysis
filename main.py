import streamlit as st
import pandas as pd
import joblib
from os import path

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Student Dropout Predictor", layout="wide")
st.title("🎓 Student Dropout Predictor")

# -----------------------------
# Load trained pipeline + target encoder
# -----------------------------
MODEL_PATH = path.join("model", "student_dropout_xgb.pkl")
pipeline, target_le = joblib.load(MODEL_PATH)

# -----------------------------
# Initialize session_state defaults
# -----------------------------
default_values = {
    "gender": "M",
    "highest_education": "No Formal quals",
    "age_band": "0-35",
    "imd_band": "0-10%",
    "disability": "N",
    "num_of_prev_attempts": 0,
    "studied_credits": 120,
    "region": "East Anglian Region",
    "code_module": "AAA",
    "code_presentation": "2013J",
    "num_assessments": 5,
    "assessments_submitted": 5,
    "avg_assessment_score": 50.0,
    "total_clicks": 100,
    "active_days": 30
}

for key, val in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -----------------------------
# Sidebar Navigation
# -----------------------------
section = st.sidebar.radio(
    "Navigate",
    ["👩‍🎓 Student Details", "📝 Assessment Details", "💻 VLE Details", "🔮 Prediction"]
)

# -----------------------------
# Predefined categories
# -----------------------------
gender_categories = ["M", "F"]
highest_education_categories = ["No Formal quals", "Lower Than A Level", "A Level or Equivalent",
                                "HE Qualification", "Post Graduate Qualification"]
age_band_categories = ["0-35", "35-55", "55<="]
imd_band_categories = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                       "50-60%", "60-70%", "70-80%", "80-90%", "90-100%", "Unknown"]
disability_categories = ["Y", "N"]
region_categories = ["East Anglian Region", "North Western Region", "Scotland", "South East Region", "Wales"]

module_codes = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
module_descriptions = {
    "AAA": "Introduction to Computing",
    "BBB": "Mathematics for Computing",
    "CCC": "Computer Networks",
    "DDD": "Database Systems",
    "EEE": "Software Engineering",
    "FFF": "Artificial Intelligence",
    "GGG": "Web Development"
}

presentation_codes = ["2013J", "2014J", "2013B", "2014B", "2015A"]
presentation_descriptions = {
    "2013J": "Oct 2013",
    "2014J": "Oct 2014",
    "2013B": "Feb 2013",
    "2014B": "Feb 2014",
    "2015A": "Aug 2015"
}

# -----------------------------
# Section: Student Details
# -----------------------------
if section == "👩‍🎓 Student Details":
    st.header("👩‍🎓 Student Details")
    with st.form("student_form"):
        st.session_state.gender = st.selectbox("Gender", gender_categories, index=gender_categories.index(st.session_state.gender))
        st.session_state.highest_education = st.selectbox("Highest Education", highest_education_categories, index=highest_education_categories.index(st.session_state.highest_education))
        st.session_state.age_band = st.selectbox("Age Band", age_band_categories, index=age_band_categories.index(st.session_state.age_band))
        st.session_state.imd_band = st.selectbox("IMD Band", imd_band_categories, index=imd_band_categories.index(st.session_state.imd_band))
        st.session_state.disability = st.selectbox("Disability", disability_categories, index=disability_categories.index(st.session_state.disability))

        st.session_state.num_of_prev_attempts = st.number_input(
            "Number of Previous Attempts (0 – 3)",
            min_value=0, max_value=3,
            value=st.session_state.num_of_prev_attempts
        )

        st.session_state.studied_credits = st.number_input(
            "Studied Credits (0 – 120)",
            min_value=0, max_value=120,
            value=st.session_state.studied_credits
        )

        st.session_state.region = st.selectbox("Region", region_categories, index=region_categories.index(st.session_state.region))
        st.session_state.code_module = st.selectbox(
            "Module",
            module_codes,
            format_func=lambda x: f"{x} - {module_descriptions.get(x)}",
            index=module_codes.index(st.session_state.code_module)
        )
        st.session_state.code_presentation = st.selectbox(
            "Presentation",
            presentation_codes,
            format_func=lambda x: f"{x} - {presentation_descriptions.get(x)}",
            index=presentation_codes.index(st.session_state.code_presentation)
        )
        st.form_submit_button("Save Student Details")

# -----------------------------
# Section: Assessment Details
# -----------------------------
elif section == "📝 Assessment Details":
    st.header("📝 Assessment Details")
    with st.form("assessment_form"):
        st.session_state.num_assessments = st.number_input(
            "Number of Assessments (0 – 20)",
            min_value=0,
            max_value=20,
            value=st.session_state.num_assessments
        )

        if st.session_state.num_assessments > 0:
            st.session_state.assessments_submitted = st.number_input(
                f"Number of Assessments Submitted (0 – {st.session_state.num_assessments})",
                min_value=0,
                max_value=st.session_state.num_assessments,
                value=st.session_state.assessments_submitted
            )
            st.session_state.avg_assessment_score = st.number_input(
                "Average Assessment Score (0 – 100)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.avg_assessment_score
            )
        else:
            st.session_state.assessments_submitted = 0
            st.session_state.avg_assessment_score = 0.0
        st.form_submit_button("Save Assessment Details")

# -----------------------------
# Section: VLE Details
# -----------------------------
elif section == "💻 VLE Details":
    st.header("💻 VLE Details")
    with st.form("vle_form"):
        st.session_state.total_clicks = st.number_input(
            "Total Clicks (0 – 100,000)",
            min_value=0,
            max_value=100000,
            value=st.session_state.total_clicks
        )

        st.session_state.active_days = st.number_input(
            "Active Days (0 – 365)",
            min_value=0,
            max_value=365,
            value=st.session_state.active_days
        )
        st.form_submit_button("Save VLE Details")

# -----------------------------
# Section: Prediction
# -----------------------------
elif section == "🔮 Prediction":
    st.header("🔮 Make Prediction")
    if st.button("Predict Outcome"):

        # --- Extreme zero-activity Withdraw ---
        if (st.session_state.num_assessments == 0 and
            st.session_state.studied_credits == 0 and
            st.session_state.total_clicks == 0 and
            st.session_state.active_days == 0):
            predicted_label = "Withdraw"
            st.success(f"Predicted Outcome: {predicted_label}")
        else:
            # --- Realistic derived features ---
            num_sites = max(1, st.session_state.active_days // 2)  # approximate unique sites visited
            derived_features = {
                "max_assessment_score": st.session_state.avg_assessment_score if st.session_state.num_assessments > 0 else 0.0,
                "min_assessment_score": st.session_state.avg_assessment_score if st.session_state.num_assessments > 0 else 0.0,
                "std_assessment_score": 0.0 if st.session_state.num_assessments <= 1 else 5.0,
                "unique_sites": num_sites,
                "avg_clicks_per_site": st.session_state.total_clicks / num_sites,
                "std_clicks": 0.0 if st.session_state.num_assessments <= 1 else 50.0,
                "no_assessments_flag": 1 if st.session_state.num_assessments == 0 else 0,
                "no_assessments": 1 if st.session_state.num_assessments == 0 else 0
            }

            # Build sample DataFrame
            sample_data = pd.DataFrame([{
                "gender": st.session_state.gender,
                "highest_education": st.session_state.highest_education,
                "imd_band": st.session_state.imd_band,
                "age_band": st.session_state.age_band,
                "num_of_prev_attempts": st.session_state.num_of_prev_attempts,
                "studied_credits": st.session_state.studied_credits,
                "disability": st.session_state.disability,
                "region": st.session_state.region,
                "code_module": st.session_state.code_module,
                "code_presentation": st.session_state.code_presentation,
                "num_assessments": st.session_state.num_assessments,
                "assessments_submitted": st.session_state.assessments_submitted,
                "avg_assessment_score": st.session_state.avg_assessment_score,
                "total_clicks": st.session_state.total_clicks,
                "active_days": st.session_state.active_days,
                **derived_features
            }])

            # Ensure all pipeline columns exist
            for col in pipeline.feature_names_in_:
                if col not in sample_data.columns:
                    sample_data[col] = 0
            sample_data = sample_data[pipeline.feature_names_in_]

            # Convert types
            numeric_cols = [c for c in sample_data.columns if c not in ["gender","highest_education","imd_band","age_band",
                                                                       "disability","region","code_module","code_presentation"]]
            sample_data[numeric_cols] = sample_data[numeric_cols].astype(float)
            cat_cols = ["gender","highest_education","imd_band","age_band",
                        "disability","region","code_module","code_presentation"]
            sample_data[cat_cols] = sample_data[cat_cols].astype(str)

            # ML Prediction
            y_pred = pipeline.predict(sample_data)
            y_pred_labels = target_le.inverse_transform(y_pred)
            st.success(f"Predicted Outcome: {y_pred_labels[0]}")


