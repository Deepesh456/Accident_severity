# Accident Severity Prediction Chatbot

An interactive Streamlit-based chatbot that predicts the severity of a road accident — **Fatal**, **Serious**, or **Slight Injury** — based on various driver, vehicle, and environmental conditions. Built using machine learning models trained on real-world accident datasets.

---

## Project Overview
---
This project is designed to assist traffic authorities, emergency response teams, and researchers in predicting accident outcomes based on reported incident details. It features a clean web-based UI, real-time predictions, and robust backend preprocessing and modeling.

---

## Features

- Real-time accident severity prediction
- Trained ML model with class balancing
- Multi-category input selection (e.g., weather, road type, driver experience)
- Streamlit interface with prediction feedback
- Model, scaler, and encoders saved using `pickle`
- Handles unseen user inputs gracefully

---

## Tech Stack

--------------------------------------------------------------
| Layer        | Tools Used                                  |
|--------------|---------------------------------------------|
| Language     | Python 3.x                                  |
| Web Framework| Streamlit                                   |
| ML Models    | Random Forest, Label Encoding, Scaler       |
| Preprocessing| Pandas, LabelEncoder, StandardScaler        |
| Deployment   | Streamlit Cloud / Localhost                 |
--------------------------------------------------------------
---

## Dataset

- **Source**: Real-world traffic and accident records (provided internally)
- **Columns**: Driver age, experience, area type, vehicle movement, collision type, weather conditions, light conditions, number of vehicles, etc.
- **Target**: `Accident_Severity` (0 = Fatal, 1 = Serious, 2 = Slight)

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/accident-severity-chatbot.git
cd accident-severity-chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## File Structure

```
├── app.py                              # Streamlit frontend
├── car_severity_prediction.csv         # Dataset
├── model.pkl                           # Trained ML model
├── scaler.pkl                          # StandardScaler for inputs
├── encoding.pkl                        # LabelEncoders for each categorical column
├── accident_severity_prediction.ipynb  # Jupyter Notebook for training
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---
## Author

**Deepeshkumar K**  
📧 deepeshkumark120@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/deepeshkumark)  
💻 [GitHub](https://github.com/Deepesh456)

---
