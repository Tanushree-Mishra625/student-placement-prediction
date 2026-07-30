# 🎓 AI Student Placement Predictor

## 📌 Project Overview

AI Student Placement Predictor is a Machine Learning based web application that predicts whether a student is likely to get placed based on academic performance, technical skills, projects, internship experience, communication skills, and other placement-related factors.

The project uses an **Artificial Neural Network (ANN)** model trained on historical student placement data and deployed as an interactive web application using **Streamlit**.

The application provides placement prediction along with confidence score, career improvement suggestions, and a downloadable PDF prediction report.

---
## 🌐 Live Demo

[AI Student Placement Predictor](https://student-placement-prediction-cqtqt2ve7qtxchxkwnpmk9.streamlit.app/)


# 🚀 Features

- ✅ Student placement prediction using Artificial Neural Network
- ✅ User-friendly interactive Streamlit interface
- ✅ Data preprocessing using feature scaling and encoding
- ✅ Placement confidence score generation
- ✅ Career improvement suggestions based on profile
- ✅ Downloadable PDF prediction report
- ✅ Clean dashboard-style UI
- ✅ Real-time prediction

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Machine Learning
- TensorFlow
- Keras
- Scikit-Learn

## Data Processing
- Pandas
- NumPy

## Visualization
- Matplotlib
- Plotly

## Web Application
- Streamlit

## Report Generation
- ReportLab

---

# 🤖 Machine Learning Model

The project uses an **Artificial Neural Network (ANN)** classification model for predicting student placement outcomes.

## Model Pipeline

```
Dataset Collection
        |
        ↓
Data Preprocessing
        |
        ↓
Feature Encoding
        |
        ↓
Feature Scaling
        |
        ↓
ANN Model Training
        |
        ↓
Model Evaluation
        |
        ↓
Streamlit Deployment
```

---

# 📂 Project Structure

```
student_placement_prediction/

│
├── app.py
│
├── placement_ann_model.keras
│
├── label_encoder.pkl
│
├── scaler.pkl
│
├── placement_prediction.ipynb
│
├── requirements.txt
│
└── README.md
```

---

# 📄 Model Files Description

### placement_ann_model.keras

Contains the trained Artificial Neural Network model used for making placement predictions.

### scaler.pkl

Stores the feature scaling object used to transform input data before prediction.

### label_encoder.pkl

Stores the encoding information for converting categorical features into numerical format.

---

# ⚙️ Installation and Setup

## 1. Clone the Repository

```bash
git clone <repository-link>
```

## 2. Navigate to Project Folder

```bash
cd student_placement_prediction
```

## 3. Create Virtual Environment

```bash
python -m venv venv
```

## 4. Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

## 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your web browser.

---

# 📊 Application Workflow

1. User enters student academic and skill-related details.
2. Input data is processed using the saved encoder and scaler.
3. The trained ANN model predicts placement probability.
4. The application displays:
   - Placement prediction result
   - Confidence score
   - Career improvement suggestions
5. User can download the prediction report as a PDF.

---

# 🎯 Key Learning Outcomes

Through this project, the following concepts were implemented:

- Data preprocessing
- Feature engineering
- Neural network model development
- Model serialization
- Machine learning deployment
- Streamlit application development
- End-to-end ML project workflow

---

# 🔮 Future Improvements

- Increase dataset size for better model performance
- Compare ANN with other ML algorithms
- Add model performance dashboard
- Deploy application on cloud platforms
- Add more personalized career recommendations

---

# 👩‍💻 Author

**Tanushree Mishra**

B.Tech Information Technology

---

⭐ If you find this project useful, consider giving it a star!