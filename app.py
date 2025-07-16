import streamlit as st
import pickle
import pandas as pd

# 💡 Add background image using CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("https://images.unsplash.com/photo-1581091012184-5c1b391c97c7?auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.set_page_config(page_title="Accident Severity Predictor", layout="centered")

    st.markdown("<h1 style='text-align: center;'>🚨 Accident Severity Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Predict the level of injury from a traffic accident based on real-world conditions.</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("🚗 Driver and Vehicle Info")
    col1, col2 = st.columns(2)

    with col1:
        Age = st.selectbox("Age band of driver", ['18-30','31-50','Under 18','Over 51'])
        Sex = st.selectbox("Sex of driver", ['Male','Female'])
        educ = st.selectbox("Education level of driver", ['Above high school','Junior high school','Elementary school','High school','Illiterate','Writing & reading'])
        rela = st.selectbox("Vehicle driver relation", ['Employee','Owner','Other'])
        exp = st.selectbox("Driving experience", ['1-2yr','Above 10yr','5-10yr','2-5yr','No Licence','Below 1yr','unknown'])
        own = st.selectbox("Owner of vehicle", ['Owner','Governmental','Organization','Other'])

    with col2:
        area = st.selectbox("Area accident occurred", ['Residential areas','Office areas','Recreational areas','Industrial areas','Church areas','Market areas','Rural village areas','Outside rural areas','Hospital areas','School areas','Other'])
        type = st.selectbox("Road surface type", ['Asphalt roads','Earth roads','Asphalt roads with some distress','Gravel roads','Other'])
        condition = st.selectbox("Road surface condition", ['Dry','Wet or damp','Snow','Flood over 3cm. deep'])
        light = st.selectbox("Light condition", ['Daylight','Darkness - lights lit','Darkness - no lighting','Darkness - lights unlit'])
        weather = st.selectbox("Weather condition", ['Normal','Raining','Raining and Windy','Cloudy','Windy','Snow','Fog or mist','Other'])
        collision = st.selectbox("Type of collision", ['Collision with roadside-parked vehicles','Vehicle with vehicle collision','Collision with roadside objects','Collision with animals','Rollover','Fall from vehicles','Collision with pedestrians','Collision with Train','Other'])

    st.markdown("---")
    st.subheader("🧍 Casualty and Movement Details")
    col3, col4 = st.columns(2)

    with col3:
        vehicles = st.selectbox("Number of vehicles involved", [1,2,3,4,5,6,7])
        casualty = st.selectbox("Number of casualties", [1,2,3,4,5,6,7,8])
        movement = st.selectbox("Vehicle movement", ['Going straight','U-Turn','Moving Backward','Turnover','Waiting to go','Getting off','Reversing','Parked','Stopping','Overtaking','Entering a junction','Other'])

    with col4:
        casuality = st.selectbox("Casualty class", ['Driver or rider','Pedestrian','Passenger'])
        sex_casuality = st.selectbox("Sex of casualty", ['Male','Female'])
        age_band = st.selectbox("Age band of casualty", ['31-50','18-30','Under 18','Over 51','5'])
        c_severity = st.selectbox("Casualty severity", ['3','2','1'])
        cause = st.selectbox("Cause of accident", ['Moving Backward','Overtaking','Changing lane to the left','Changing lane to the right','Overloading','No priority to vehicle','No priority to pedestrian','No distancing','Getting off the vehicle improperly','Improper parking','Overspeed','Driving carelessly','Driving at high speed','Driving to the left','Overturning','Turnover','Driving under the influence of drugs','Drunk driving'])

    df = pd.DataFrame([[Age, Sex, educ, rela, exp, own, area, type, condition, light, weather, collision,
                        vehicles, casualty, movement, casuality, sex_casuality, age_band, c_severity, cause]])

    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    label = pickle.load(open('encoding.pkl', 'rb'))

    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in object_columns:
        df[col] = label.fit_transform(df[col])

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    pred = st.button("Predict Accident Severity")

    if pred:
        prediction = model.predict(scaler.transform(df))
        if prediction == 0:
            st.error("☠️ **Fatal Injury** — Immediate emergency required.")
        elif prediction == 1:
            st.warning("🚑 **Serious Injury** — Emergency attention advised.")
        else:
            st.success("🩹 **Slight Injury** — Minor injury reported.")

main()
