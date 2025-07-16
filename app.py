import streamlit as st
import pickle
import pandas as pd
from PIL import Image

# Load model components
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoding.pkl', 'rb'))  # OrdinalEncoder recommended

st.set_page_config(page_title="Accident Severity Predictor", layout="centered")
st.title("🚨 Accident Severity Prediction Chatbot")
st.markdown("Predict the likely **severity** of an accident based on conditions and inputs.")

# Sidebar Image
with st.sidebar:
    st.image("accident.jpg", use_column_width=True)
    st.markdown("### Designed by Deepeshkumar K")
    st.markdown("[GitHub](https://github.com/Deepesh456) | [LinkedIn](https://linkedin.com/in/deepeshkumark)")

st.markdown("---")
st.subheader("📝 Accident Details")

# Create layout columns
col1, col2 = st.columns(2)

with col1:
    Age = st.selectbox("👤 Age band of driver", ['Under 18', '18-30', '31-50', 'Over 51'])
    Sex = st.selectbox("🚻 Sex of driver", ['Male', 'Female'])
    educ = st.selectbox("🎓 Education level", ['Illiterate', 'Elementary school', 'Junior high school', 'High school', 'Above high school', 'Writing & reading'])
    exp = st.selectbox("🧭 Driving experience", ['No Licence', 'Below 1yr', '1-2yr', '2-5yr', '5-10yr', 'Above 10yr', 'unknown'])
    own = st.selectbox("🚗 Vehicle Owner", ['Owner', 'Governmental', 'Organization', 'Other'])
    area = st.selectbox("📍 Accident Area", ['Residential areas','Office areas','Recreational areas','Industrial areas','Church areas','Market areas','Rural village areas','Outside rural areas','Hospital areas','School areas','Other'])

with col2:
    rela = st.selectbox("👥 Driver-Vehicle Relation", ['Employee', 'Owner', 'Other'])
    type = st.selectbox("🛣️ Road Surface Type", ['Asphalt roads','Earth roads','Asphalt roads with some distress','Gravel roads','Other'])
    condition = st.selectbox("🌧️ Road Surface Condition", ['Dry','Wet or damp','Snow','Flood over 3cm. deep'])
    light = st.selectbox("💡 Light Condition", ['Daylight','Darkness - lights lit','Darkness - no lighting','Darkness - lights unlit'])
    weather = st.selectbox("☁️ Weather", ['Normal','Raining','Raining and Windy','Cloudy','Windy','Snow','Fog or mist','Other'])
    collision = st.selectbox("💥 Collision Type", ['Collision with roadside-parked vehicles','Vehicle with vehicle collision','Collision with roadside objects','Collision with animals','Rollover','Fall from vehicles','Collision with pedestrians','Collision with Train','Other'])

st.markdown("---")
st.subheader("👨‍⚕️ Casualty Details")

col3, col4 = st.columns(2)
with col3:
    vehicles = st.selectbox("🚘 Number of Vehicles Involved", [1,2,3,4,5,6,7])
    casualty = st.selectbox("🚑 Number of Casualties", [1,2,3,4,5,6,7,8])
    movement = st.selectbox("➡️ Vehicle Movement", ['Going straight','U-Turn','Moving Backward','Turnover','Waiting to go','Getting off','Reversing','Parked','Stopping','Overtaking','Entering a junction','Other'])

with col4:
    casuality = st.selectbox("🧍 Casualty Class", ['Driver or rider','Pedestrian','Passenger'])
    sex_casuality = st.selectbox("🚻 Sex of Casualty", ['Male','Female'])
    age_band = st.selectbox("👶 Age Band of Casualty", ['Under 18','18-30','31-50','Over 51','5'])  # "5" seems off — consider fixing in model
    c_severity = st.selectbox("❗ Casualty Severity (Ground Truth)", ['3','2','1'])
    cause = st.selectbox("⚠️ Cause of Accident", ['Moving Backward','Overtaking','Changing lane to the left','Changing lane to the right','Overloading','No priority to vehicle','No priority to pedestrian','No distancing','Getting off the vehicle improperly','Improper parking','Overspeed','Driving carelessly','Driving at high speed','Driving to the left','Overturning','Turnover','Driving under the influence of drugs','Drunk driving'])

# Assemble DataFrame
df = pd.DataFrame([[Age, Sex, educ, rela, exp, own, area, type, condition, light, weather, collision,
                    vehicles, casualty, movement, casuality, sex_casuality, age_band, c_severity, cause]],
                  columns=['Age', 'Sex', 'Education', 'Relation', 'Experience', 'Ownership', 'Area', 'Surface', 'Surface_Condition', 
                           'Light', 'Weather', 'Collision', 'Vehicles', 'Casualties', 'Movement', 'Casualty_Class', 
                           'Sex_Casualty', 'Age_Casualty', 'Casualty_Severity', 'Cause'])

# Encode
cat_cols = df.select_dtypes(include='object').columns.tolist()
try:
    df[cat_cols] = encoder.transform(df[cat_cols])
except ValueError as e:
    st.error("🚫 Invalid input — contains unseen label. Please check for typos or unexpected values.")
    st.stop()

# Predict
if st.button("🔍 Predict Severity"):
    try:
        prediction = model.predict(scaler.transform(df))[0]
        if prediction == 0:
            st.success("☠️ **Fatal Injury** — Immediate action required!")
        elif prediction == 1:
            st.warning("🚑 **Serious Injury** — Emergency response needed.")
        else:
            st.info("🩹 **Slight Injury** — Minor accident.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
