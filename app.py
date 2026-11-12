

from pathlib import Path
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="EV Charging Demand Prediction",
    page_icon="⚡",
    layout="wide"
)

MODEL_DIR = Path(r"C:\Users\Lenovo")
PROJECT_DIR = Path(r"C:\Users\Lenovo\EV_Charging_Dashboard")

# --------------------------------------------------
# 2. CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>

/* Main dashboard background */
.stApp {
    background: linear-gradient(
        135deg,
        #071B4A 0%,
        #123B78 45%,
        #176B87 100%
    );
    color: #FFFFFF;
}

/* Header */
[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #101D50 0%,
        #263B78 55%,
        #176B87 100%
    );
    min-width: 290px !important;
    max-width: 290px !important;
}

/* Text */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp p,
.stApp label,
.stApp span {
    color: #FFFFFF;
}

/* Navigation text */
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #FFFFFF !important;
    white-space: nowrap !important;
    font-size: 0.88rem !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #243F82, #087E9B);
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #55D6E8;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #7B2FF7, #00C6D7);
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: bold;
}

.stButton > button p,
.stFormSubmitButton > button p {
    color: white !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 8px;
}

/* Divider */
hr {
    border-color: #55D6E8;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD TRAINED MODELS AND RESULTS
# --------------------------------------------------

@st.cache_resource
def load_pickle(filename):
    with open(MODEL_DIR / filename, "rb") as file:
        return pickle.load(file)


@st.cache_data
def load_results(filename):
    return pd.read_csv(MODEL_DIR / filename)


try:
    classifier = load_pickle("ev_charging_xgb_classifier.pkl")
    regressor = load_pickle("ev_charging_xgb_regressor.pkl")

    classification_features = load_pickle(
        "ev_charging_feature_columns.pkl"
    )

    regression_features = load_pickle(
        "ev_charging_regression_feature_columns.pkl"
    )

    classification_results = load_results(
        "ev_charging_classification_comparison.csv"
    )

    regression_results = load_results(
        "ev_charging_regression_comparison.csv"
    )

except Exception as error:
    st.error(f"Unable to load a project file: {error}")
    st.stop()


# --------------------------------------------------
# 4. DASHBOARD HEADER
# --------------------------------------------------

st.title("⚡ EV Charging Demand Prediction")

st.markdown(
    "A machine learning dashboard for exploring EV charging demand, "
    "comparing model performance, and generating predictions."
)

st.divider()


# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Project Overview",
        "Classification Performance",
        "Regression Performance",
        "Demand Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Models: XGBoost Classifier and XGBoost Regressor"
)


# --------------------------------------------------
# 6. PROJECT OVERVIEW
# --------------------------------------------------

if page == "Project Overview":

    st.header("Project Overview")

    st.write(
        "This project uses two trained machine learning models. "
        "The classifier predicts a demand category, while the "
        "regressor estimates a numerical demand value."
    )

    col1, col2, col3 = st.columns(3)

    best_classification = classification_results.loc[
        classification_results["F1 Score"].idxmax()
    ]

    best_regression = regression_results.loc[
        regression_results["R2 Score"].idxmax()
    ]

    col1.metric(
        "Best Classification Model",
        str(best_classification["Model"])
    )

    col2.metric(
        "Classification F1 Score",
        f'{best_classification["F1 Score"]:.3f}'
    )

    col3.metric(
        "Regression R² Score",
        f'{best_regression["R2 Score"]:.3f}'
    )

    st.subheader("What does each model do?")

    st.markdown("""
    - **Classification:** Predicts which demand category an observation belongs to.
    - **Regression:** Estimates a numerical demand value.
    - **Model comparison:** Compares several algorithms using evaluation metrics.
    """)

    st.subheader("Project Workflow")

    st.markdown("""
    1. Prepare the EV charging dataset.
    2. Train classification and regression models.
    3. Evaluate model performance.
    4. Load the trained XGBoost models.
    5. Use the dashboard to explore results and make predictions.
    """)


# --------------------------------------------------
# 7. CLASSIFICATION PERFORMANCE
# --------------------------------------------------

elif page == "Classification Performance":

    st.header("Classification Model Performance")

    st.write(
        "Compare the classification algorithms using Accuracy, "
        "Precision, Recall, F1 Score and ROC-AUC."
    )

    st.dataframe(
        classification_results.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    metric = st.selectbox(
        "Choose a metric to compare",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    )

    chart = px.bar(
        classification_results,
        x="Model",
        y=metric,
        color="Model",
        title=f"Classification Models — {metric}",
        text_auto=".3f"
    )

    chart.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#172B4D"
    )

    st.plotly_chart(chart, use_container_width=True)


# --------------------------------------------------
# 8. REGRESSION PERFORMANCE
# --------------------------------------------------

elif page == "Regression Performance":

    st.header("Regression Model Performance")

    st.write(
        "Compare regression models. Lower MAE and RMSE indicate "
        "smaller prediction errors; higher R² indicates better "
        "explained variation on the evaluation data."
    )

    st.dataframe(
        regression_results.style.format({
            "MAE": "{:.3f}",
            "RMSE": "{:.3f}",
            "R2 Score": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    metric = st.selectbox(
        "Choose a regression metric",
        ["MAE", "RMSE", "R2 Score"]
    )

    chart = px.bar(
        regression_results,
        x="Model",
        y=metric,
        color="Model",
        title=f"Regression Models — {metric}",
        text_auto=".3f"
    )

    chart.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#172B4D"
    )

    st.plotly_chart(chart, use_container_width=True)


# --------------------------------------------------
# 9. DEMAND PREDICTION
# --------------------------------------------------

elif page == "Demand Prediction":

    st.header("EV Charging Demand Prediction")

    st.write(
        "Enter the charging session details below to predict "
        "the demand category and numerical demand."
    )

    locations = [
        "accommodation",
        "apartment",
        "bus garage",
        "camping",
        "company",
        "golf",
        "hotel",
        "market",
        "public area",
        "public institution",
        "public parking lot",
        "resort",
        "restaurant",
        "sightseeing"
    ]

    days = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    months = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    time_periods = [
        "Night", "Morning", "Afternoon", "Evening"
    ]

    # Actual class mapping confirmed from the training data:
    # 0 = Normal Demand
    # 1 = High Demand
    demand_labels = {
        0: "Normal Demand",
        1: "High Demand"
    }
    # --------------------------------------------------
    # 9.1 USER INPUT FORM
    # --------------------------------------------------

    with st.form("prediction_form"):

        st.subheader("Charging Session Details")

        left_col, right_col = st.columns(2, gap="large")

        # LEFT COLUMN
        with left_col:

            day_label = st.selectbox(
                "Day of Week",
                days
            )

            month_label = st.selectbox(
                "Month",
                months
            )

            hour = st.selectbox(
                "Hour",
                list(range(24)),
                format_func=lambda x: f"{x:02d}:00"

            )
            # Automatically determine Time of Day from Hour
            if 0 <= hour <= 5:
                time_of_day = "Night"
            elif 6 <= hour <= 11:
                time_of_day = "Morning"
            elif 12 <= hour <= 17:
                time_of_day = "Afternoon"
            else:
                time_of_day = "Evening"

            st.text_input(
                "Time of Day",
                 value=time_of_day,
                disabled=True
)


        # RIGHT COLUMN
        with right_col:

            duration = st.number_input(
                "Charging Duration",
                min_value=1.0,
                max_value=1573.0,
                value=60.0,
                step=0.5
            )


            # Charger details displayed one below the other
            charger_company = st.selectbox(
                "Charger Company",
                [0, 1],
                format_func=lambda x: f"Company {x}"
            )

            charger_type = st.selectbox(
                "Charger Type",
                [0, 1],
                format_func=lambda x: f"Type {x}"
            )
            location = st.selectbox(
                "Location",
                locations
            )

        submitted = st.form_submit_button(
            "Predict Demand",
            use_container_width=True
        )

    # --------------------------------------------------
    # 10. PROCESS INPUTS AND GENERATE PREDICTIONS
    # --------------------------------------------------

    if submitted:

        # Convert selections to numerical values.
        day_number = days.index(day_label)
        month_number = months.index(month_label) + 1

        # Automatically derive weekend status from the selected day.
        # Monday-Friday = 0 (weekday); Saturday-Sunday = 1 (weekend).
        is_weekend = 1 if day_label in ["Saturday", "Sunday"] else 0

        # Prepare the selected input values.
        raw_values = {
            "ChargerCompany": charger_company,
            "ChargerType": charger_type,
            "Duration": duration,
            "Hour": hour,
            "DayOfWeek": day_number,
            "Month": month_number,
            "IsWeekend": is_weekend
        }

        # Encode Time of Day
        for period in ["Morning", "Afternoon", "Evening"]:
            raw_values[f"TimeOfDay_{period}"] = (
                1 if time_of_day == period else 0
            )

        # Encode location using saved model feature names.
        for loc in locations:

            feature_name = f"Location_{loc}"

            raw_values[feature_name] = (
                1 if location == loc else 0
            )

        # Encode Time of Day.
        # Night is assumed to be the reference category.
        for period in ["Morning", "Afternoon", "Evening"]:

            raw_values[f"TimeOfDay_{period}"] = (
                1 if time_of_day == period else 0
            )

        # Create classification input in saved feature order.
        classification_input = pd.DataFrame(
            [[raw_values.get(feature, 0)
              for feature in classification_features]],
            columns=classification_features
        )

        # Create regression input in saved feature order.
        regression_input = pd.DataFrame(
            [[raw_values.get(feature, 0)
              for feature in regression_features]],
            columns=regression_features
        )

        st.divider()
        st.subheader("Prediction Results")
        st.caption(
            f"Selected day: {day_label} | Weekend status: {'Yes' if is_weekend else 'No'}"
        )

        # --------------------------------------------------
        # 10.1 CLASSIFICATION PREDICTION
        # --------------------------------------------------

        try:

            predicted_class = classifier.predict(
                classification_input
            )[0]

            # Display readable category names.
            predicted_label = demand_labels.get(
                predicted_class,
                str(predicted_class)
            )

            st.metric(
                "Predicted Demand Category",
                predicted_label
            )

            # Show probabilities for each trained class.
            if hasattr(classifier, "predict_proba"):

                probabilities = classifier.predict_proba(
                    classification_input
                )[0]

                probability_df = pd.DataFrame({
                    "Demand Category": [
                        demand_labels.get(label, str(label))
                        for label in classifier.classes_
                    ],
                    "Probability (%)": (
                        probabilities * 100
                    ).round(2)
                })

                st.write("Prediction Probabilities")

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:

            st.error(
                f"Classification prediction failed: {error}"
            )

        # --------------------------------------------------
        # 10.2 REGRESSION PREDICTION
        # --------------------------------------------------

        try:

            predicted_value = regressor.predict(
                regression_input
            )[0]

            st.metric(
                "Predicted Numerical Demand",
                f"{predicted_value:.3f}"
            )

        except Exception as error:

            st.error(
                f"Regression prediction failed: {error}"
            )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "EV Charging Demand Prediction | Machine Learning Project"
)
