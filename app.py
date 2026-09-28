import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD TRAINED MODELS
# =========================================================

crop_model = joblib.load("models/crop_model.pkl")
price_model = joblib.load("models/price_model.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Crop Recommendation & Price Prediction",
    page_icon="🌾",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🌾 AI-Based Crop Recommendation and Price Prediction System")

st.write(
    "A Machine Learning based web application that recommends "
    "suitable crops and predicts crop market prices."
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🌾 About the Project")

st.sidebar.write(
    "This project uses Machine Learning techniques to support "
    "farmers in crop selection and market price estimation."
)

st.sidebar.subheader("Technologies Used")

st.sidebar.write("""
- Python
- Pandas
- Scikit-learn
- Random Forest
- Joblib
- Streamlit
""")


# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs([
    "🌱 Crop Recommendation",
    "💰 Price Prediction"
])


# =========================================================
# CROP RECOMMENDATION
# =========================================================

with tab1:

    st.header("🌱 Crop Recommendation")

    st.write(
        "Enter the soil and environmental conditions "
        "to get a suitable crop recommendation."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            value=90.0
        )

    with col2:
        P = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            value=42.0
        )

    with col3:
        K = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            value=43.0
        )

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        temperature = st.number_input(
            "Temperature (°C)",
            value=21.0
        )

    with col5:
        humidity = st.number_input(
            "Humidity (%)",
            value=82.0
        )

    with col6:
        ph = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5
        )

    with col7:
        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=203.0
        )

    st.write("")

    if st.button("🌱 Recommend Crop", use_container_width=True):

        input_data = pd.DataFrame([{
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }])

        prediction = crop_model.predict(input_data)

        st.success(
            f"Recommended Crop: {prediction[0].capitalize()} 🌾"
        )


# =========================================================
# PRICE PREDICTION
# =========================================================

with tab2:

    st.header("💰 Crop Price Prediction")

    st.write(
        "Enter the market and crop details to estimate "
        "the modal market price."
    )

    col1, col2 = st.columns(2)

    with col1:

        state = st.text_input(
            "State",
            "Andhra Pradesh"
        )

        district = st.text_input(
            "District",
            "Chittor"
        )

        market = st.text_input(
            "Market",
            "Chittoor"
        )

        commodity = st.text_input(
            "Commodity",
            "Rice"
        )

    with col2:

        variety = st.text_input(
            "Variety",
            "Common"
        )

        grade = st.text_input(
            "Grade",
            "FAQ"
        )

        min_price = st.number_input(
            "Minimum Price (₹)",
            min_value=0.0,
            value=3200.0
        )

        max_price = st.number_input(
            "Maximum Price (₹)",
            min_value=0.0,
            value=3400.0
        )

    st.write("")

    if st.button("💰 Predict Price", use_container_width=True):

        price_input = pd.DataFrame([{
            "State": state,
            "District": district,
            "Market": market,
            "Commodity": commodity,
            "Variety": variety,
            "Grade": grade,
            "Min Price": min_price,
            "Max Price": max_price
        }])

        price_prediction = price_model.predict(price_input)

        st.success(
            f"Predicted Modal Price: ₹{price_prediction[0]:,.2f}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Developed using Python, Machine Learning and Streamlit"
)