import streamlit as st
import pickle
import pandas as pd

from PIL import Image
def main():
    st.title("Accident Severity Prediction")
    img=Image.open("accident.jpg")
    st.image(img)
    Age=st.selectbox("Age band of driver",['18-30','31-50','Under 18','Over 51'])
    Sex=st.selectbox("Sex of driver",['Male','Female'])
    educ=st.selectbox("Education level of driver",['Above high school','Junior high school','Elementary school','High school','Illiterate','Writing & reading'])
    rela=st.selectbox("Vehicle driver relation",['Employee','Owner','Other'])
    exp=st.selectbox("Driving experience",['1-2yr','Above 10yr','5-10yr','2-5yr','No Licence','Below 1yr','unknown'])
    own=st.selectbox("Owner of vehicle",['Owner','Governmental','Organization','Other'])
    area=st.selectbox("Area accident occured",['Residential areas','Office areas','Recreational areas','Industrial areas','Church areas','Market areas','Rural village areas','Outside rural areas','Hospital areas','School areas','Rural village areas','Other'])
    type=st.selectbox("Road surface type",['Asphalt roads','Earth roads','Asphalt roads with some distress','Gravel roads','Other'])
    condition=st.selectbox("Road surface conditions",['Dry','Wet or damp','Snow','Flood over 3cm. deep'])
    light=st.selectbox("Light conditions",['Daylight','Darkness - lights lit','Darkness - no lighting','Darkness - lights unlit'])
    weather=st.selectbox("Weather conditions",['Normal','Raining','Raining and Windy','Cloudy','Windy','Snow','Fog or mist','Other'])
    collision=st.selectbox("Type of collision",['Collision with roadside-parked vehicles','Vehicle with vehicle collision','Collision with roadside objects','Collision with animals','Rollover','Fall from vehicles','Collision with pedestrians','Collision with Train','Other'])
    vehicles=st.selectbox("Number of vehicles involved",[1,2,3,4,5,6,7])
    casualty=st.selectbox("Number of casualties",[1,2,3,4,5,6,7,8])
    movement=st.selectbox("Vehicle movement",['Going straight','U-Turn','Moving Backward','Turnover','Waiting to go','Getting off','Reversing','Parked','Stopping','Overtaking','Entering a junction','Other'])
    casuality=st.selectbox("Casualty class",['Driver or rider','Pedestrian','Passenger'])
    sex_casuality=st.selectbox("Sex of casualty",['Male','Female'])
    age_band=st.selectbox("Age band of casualty",['31-50','18-30','Under 18','Over 51','5'])
    c_severity=st.selectbox("Casualty severity",['3','2','1'])
    cause=st.selectbox("Cause of accident",['Moving Backward','Overtaking','Changing lane to the left','Changing lane to the right','Overloading','No priority to vehicle','No priority to pedestrian','No distancing','Getting off the vehicle improperly','Improper parking','Overspeed','Driving carelessly','Driving at high speed','Driving to the left','Overturning','Turnover','Driving under the influence of drugs','Drunk driving'])


    df=pd.DataFrame([[Age,Sex,educ,rela,exp,own,area,type,condition,light,weather,collision,vehicles,casualty,movement,casuality,sex_casuality,age_band,c_severity,cause]])

    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    label = pickle.load(open('encoding.pkl', 'rb'))

    object_columns=df.select_dtypes(include=['object']).columns.tolist()
    for col in object_columns:
        df[col]=label.fit_transform(df[col])

    col=df.columns

    pred = st.button('PREDICT')

    if pred:
        prediction = model.predict(scaler.transform(df))
        if prediction == 0:
            st.write("Fatal Injury")
        elif prediction == 1:
            st.write("Serious Injury")
        else:
            st.write("Slight Injury")

main()
