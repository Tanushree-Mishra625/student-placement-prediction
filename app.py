# ============================
# IMPORT LIBRARIES
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

from reportlab.lib import colors as report_colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from io import BytesIO


# ============================
# PAGE CONFIGURATION
# ============================

st.set_page_config(
    page_title="AI Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================
# COLOR PALETTE
# ============================

PRIMARY = "#3B82F6"
SUCCESS = "#22C55E"
DANGER = "#EF4444"
PURPLE = "#8B5CF6"
CARD = "#1E293B"
BACKGROUND = "#0F172A"
TEXT = "#F8FAFC"
SUBTEXT = "#CBD5E1"


# ============================
# CUSTOM CSS
# ============================


st.markdown("""
<style>

/* ================================
   MAIN BACKGROUND
================================ */

.stApp{
    background-color:#0F172A;
}


/* ================================
   REMOVE EXTRA TOP SPACE
================================ */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}


/* ================================
   SIDEBAR
================================ */

[data-testid="stSidebar"]{
    background:#111827;
}


/* Sidebar text */

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] span{
    color:white;
}


/* ================================
   HERO BANNER
================================ */

.hero{

    background:linear-gradient(135deg,#2563EB,#7C3AED);

    padding:40px;

    border-radius:18px;

    text-align:center;

    color:white;

    margin-bottom:30px;

    box-shadow:0px 8px 25px rgba(0,0,0,0.30);

}


.hero h1{

    font-size:42px;

    margin-bottom:10px;

}


.hero p{

    font-size:18px;

}


/* ================================
   CARDS
================================ */

.card{

    background:#1E293B;

    border-radius:18px;

    padding:25px;

    border:1px solid #334155;

    text-align:center;

    min-height:210px;

    box-shadow:0px 6px 18px rgba(0,0,0,0.25);

    transition:0.3s;

}


.card:hover{

    transform:translateY(-6px);

    border:1px solid #3B82F6;

}


.card h2{

    font-size:42px;

}


.card h3{

    color:white;

}


.card p{

    color:#CBD5E1;

}



/* ================================
   METRIC CARDS
================================ */

div[data-testid="metric-container"]{

    background:#1E293B;

    border:1px solid #334155;

    padding:18px;

    border-radius:15px;

}



/* ================================
   BUTTONS
================================ */

.stButton button{

    width:100%;

    height:55px;

    border-radius:12px;

    border:none;

    background:#2563EB;

    color:white;

    font-size:18px;

    font-weight:bold;

    transition:0.3s;

}


.stButton button:hover{

    background:#1D4ED8;

}



/* ================================
   DOWNLOAD BUTTON
================================ */

.stDownloadButton button{

    width:100%;

    height:50px;

    border-radius:12px;

}



/* ================================
   SUCCESS MESSAGE BOX
================================ */

.success-box{

    background:#16A34A;

    padding:20px;

    border-radius:15px;

    color:white;

    text-align:center;

}



/* ================================
   ERROR MESSAGE BOX
================================ */

.error-box{

    background:#DC2626;

    padding:20px;

    border-radius:15px;

    color:white;

    text-align:center;

}



/* ================================
   FOOTER
================================ */

.footer{

    text-align:center;

    color:#94A3B8;

    margin-top:60px;

    font-size:15px;

}



/* ================================
   HORIZONTAL LINE
================================ */

hr{

    border:1px solid #334155;

}



/* ================================
   HIDE STREAMLIT BRANDING ONLY
================================ */

#MainMenu{

    display:none;

}


footer{

    display:none;

}


/*
DO NOT HIDE HEADER
because it contains sidebar toggle
*/

header{

    background:transparent;

}


</style>
""", unsafe_allow_html=True)

# ============================
# LOAD MODEL
# ============================

model = tf.keras.models.load_model("placement_ann_model.keras")
scaler = joblib.load("scaler.pkl")

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        """
        <h2 style='text-align:center;'>🎓</h2>
        <h2 style='text-align:center;margin-top:-15px;'>
        AI Placement Predictor
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;color:lightgray;'>Artificial Neural Network</p>",
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🎯 Predict Placement"
        ]
    )

    st.divider()

    st.markdown("### 📊 Model Information")

    st.metric("Model", "ANN")

    st.metric("Dataset", "10,000 Students")

    st.metric("Problem", "Classification")

    st.divider()

    st.info(
        """
This application predicts whether a student is
likely to be placed based on academic and
technical performance.
"""
    )



# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="hero">

        <h1>🎓 AI Student Placement Predictor</h1>

        <h3>Predict Campus Placement using Artificial Intelligence</h3>

        <p>
        An interactive machine learning application that predicts
        student placement using an Artificial Neural Network (ANN).
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ======================================
    # FEATURES
    # ======================================

    st.subheader("✨ Key Features")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            """
            <div class="card">

            <h2>🤖</h2>

            <h3>AI Prediction</h3>

            <p>
            Predict placement instantly using
            Artificial Neural Network.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">

            <h2>📊</h2>

            <h3>Interactive Charts</h3>

            <p>
            Visualize prediction probability
            and student profile.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            """
            <div class="card">

            <h2>💡</h2>

            <h3>Career Suggestions</h3>

            <p>
            Get personalized suggestions
            for improving placement chances.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            """
            <div class="card">

            <h2>📄</h2>

            <h3>PDF Report</h3>

            <p>
            Download your complete
            prediction report instantly.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )



    st.write("")
    st.write("")



    # ======================================
    # MODEL OVERVIEW
    # ======================================

    st.subheader("📈 Model Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset", "10,000")

    with col2:
        st.metric("Features", "11")

    with col3:
        st.metric("Model", "ANN")

    with col4:
        st.metric("Task", "Binary Classification")



    st.write("")
    st.write("")



    # ======================================
    # TECHNOLOGY STACK
    # ======================================

    st.subheader("🛠 Technology Stack")

    t1, t2, t3 = st.columns(3)

    with t1:

        st.success("🐍 Python")

        st.success("📊 Pandas")

        st.success("🔢 NumPy")

    with t2:

        st.success("🤖 TensorFlow")

        st.success("🧠 Keras")

        st.success("⚙️ Scikit-Learn")

    with t3:

        st.success("🎨 Streamlit")

        st.success("📄 ReportLab")

        st.success("📈 Matplotlib")



    st.write("")
    st.write("")



    # ======================================
    # HOW IT WORKS
    # ======================================

    st.subheader("⚙️ How It Works")

    st.markdown(
        """
1️⃣ Enter student academic and technical details.

2️⃣ Data is preprocessed using the saved StandardScaler.

3️⃣ The trained ANN model predicts placement probability.

4️⃣ The application displays:

- Placement Status
- Confidence Score
- Charts
- Career Suggestions

5️⃣ Download a professional PDF report.
"""
    )



    st.write("")
    st.write("")



    # ======================================
    # ABOUT PROJECT
    # ======================================

    st.subheader("📚 About This Project")

    st.write(
        """
This project predicts whether a student is likely to be placed based on
important academic and technical attributes such as CGPA, projects,
internships, communication skills, hackathon participation,
and backlogs.

The prediction model is built using an Artificial Neural Network (ANN)
trained on historical student placement data.

The goal of this project is to help students understand their placement
readiness and identify areas for improvement.
"""
    )



    st.write("")
    st.write("")



    # ======================================
    # FOOTER
    # ======================================

    st.markdown(
        """
        <div class="footer">

        <hr>

        Developed as a B.Tech Major Project

        <br><br>

        <b>Technologies Used</b>

        <br>

        Python • TensorFlow • Streamlit • Scikit-Learn

        <br><br>

        © 2026 Student Placement Prediction System

        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
# PREDICTION PAGE
# ==========================================

elif page == "🎯 Predict Placement":

    st.markdown("""
    <div class="hero">
        <h1>🎯 Predict Placement</h1>
        <p>Enter the student's academic and technical details below.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.subheader("📝 Student Information")

    with st.container():

        col1, col2 = st.columns(2)

        # -----------------------------
        # LEFT COLUMN
        # -----------------------------
        with col1:

            st.markdown("#### 🎓 Academic Details")

            cgpa = st.slider(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.1
            )

            tenth = st.slider(
                "10th Percentage",
                0,
                100,
                75
            )

            twelfth = st.slider(
                "12th Percentage",
                0,
                100,
                75
            )

            communication = st.slider(
                "Communication Skill Rating",
                1.0,
                5.0,
                4.0,
                step=0.1
            )

            backlogs = st.number_input(
                "Current Backlogs",
                min_value=0,
                max_value=10,
                value=0
            )

        # -----------------------------
        # RIGHT COLUMN
        # -----------------------------
        with col2:

            st.markdown("#### 💻 Technical Details")

            skills = st.slider(
                "Skills Rating",
                0,
                10,
                7
            )

            major_projects = st.selectbox(
                "Major Projects",
                [0, 1, 2, 3]
            )

            mini_projects = st.selectbox(
                "Mini Projects",
                [0, 1, 2, 3, 4]
            )

            workshops = st.selectbox(
                "Workshops / Certifications",
                [0, 1, 2, 3, 4, 5]
            )

            internship = st.selectbox(
                "Internship",
                ["No", "Yes"]
            )

            hackathon = st.selectbox(
                "Hackathon",
                ["No", "Yes"]
            )

    st.write("")
    st.write("")

    st.markdown("---")

    c1, c2, c3 = st.columns([1,2,1])

    with c2:

        predict_btn = st.button(
            "🚀 Predict Placement",
            use_container_width=True
        )

    st.write("")

    # ==========================================
    # MODEL PREDICTION
    # ==========================================

    if predict_btn:

        with st.spinner("🤖 AI is analysing the student's profile..."):

            # -----------------------------------
            # Convert categorical values
            # -----------------------------------

            internship_value = 1 if internship == "Yes" else 0
            hackathon_value = 1 if hackathon == "Yes" else 0

            # -----------------------------------
            # Create Feature Vector
            # -----------------------------------

            features = np.array([[
                cgpa,
                major_projects,
                workshops,
                mini_projects,
                skills,
                communication,
                internship_value,
                hackathon_value,
                twelfth,
                tenth,
                backlogs
            ]])

            # -----------------------------------
            # Scale Features
            # -----------------------------------

            scaled_features = scaler.transform(features)

            # -----------------------------------
            # ANN Prediction
            # -----------------------------------

            probability = float(
                model.predict(
                    scaled_features,
                    verbose=0
                )[0][0]
            )

            prediction = 1 if probability >= 0.5 else 0

            if prediction == 1:
                confidence = probability * 100
            else:
                confidence = (1 - probability) * 100

            # -----------------------------------
            # Prediction Result
            # -----------------------------------

            st.write("")
            st.markdown("---")
            st.write("")

            if prediction == 1:

                st.markdown(f"""
                <div class="success-box">

                <h1>🎉 PLACED</h1>

                <h3>Placement Probability</h3>

                <h2>{confidence:.2f}%</h2>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="error-box">

                <h1>❌ NOT PLACED</h1>

                <h3>Placement Probability</h3>

                <h2>{confidence:.2f}%</h2>

                </div>
                """, unsafe_allow_html=True)

            # -----------------------------------
            # Confidence Meter
            # -----------------------------------

            st.write("")

            st.subheader("📊 Prediction Confidence")

            st.progress(int(confidence))

            st.write(
                f"**Confidence Score :** {confidence:.2f}%"
            )

            st.write("")

            # ==========================================
            # STUDENT ANALYSIS DASHBOARD
            # ==========================================

            st.markdown("---")

            st.subheader("📈 Student Performance Dashboard")

            chart_col1, chart_col2 = st.columns(2)

            # ==========================================
            # CHART 1 : Placement Probability
            # ==========================================

            with chart_col1:

                fig, ax = plt.subplots(figsize=(5,5))

                probability_values = [
                    probability,
                    1 - probability
                ]

                labels = [
                    "Placed",
                    "Not Placed"
                ]

                colors = [
                    "#22C55E",
                    "#EF4444"
                ]

                ax.pie(
                    probability_values,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=colors
                )

                ax.set_title("Placement Probability")

                st.pyplot(fig)



            # ==========================================
            # CHART 2 : Student Profile
            # ==========================================

            with chart_col2:

                profile = {

                    "CGPA": cgpa,

                    "Skills": skills,

                    "Communication": communication * 2,

                    "Projects": (major_projects + mini_projects) * 2,

                    "Academics": (tenth + twelfth) / 20

                }

                fig2, ax2 = plt.subplots(figsize=(6,4))

                ax2.bar(
                    profile.keys(),
                    profile.values(),
                    color=[
                        "#3B82F6",
                        "#22C55E",
                        "#8B5CF6",
                        "#F59E0B",
                        "#EC4899"
                    ]
                )

                ax2.set_ylim(0,10)

                ax2.set_ylabel("Score")

                ax2.set_title("Student Profile Analysis")

                st.pyplot(fig2)



            # ==========================================
            # SUMMARY
            # ==========================================

            st.markdown("---")

            st.subheader("📋 Student Summary")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:

                st.success(f"""
            CGPA : {cgpa}

            Skills : {skills}/10

            Communication : {communication}/5

            Projects : {major_projects + mini_projects}

            Internship : {internship}
            """)

            with summary_col2:

                st.info(f"""
            Hackathon : {hackathon}

            Backlogs : {backlogs}

            10th % : {tenth}

            12th % : {twelfth}

            Confidence : {confidence:.2f}%
            """)



            # ==========================================
            # CAREER SUGGESTIONS
            # ==========================================

            st.markdown("---")

            st.subheader("💡 Personalized Suggestions")

            suggestions = []

            if cgpa < 7:
                suggestions.append(
                    "📘 Improve your CGPA to increase placement opportunities."
                )

            if skills < 7:
                suggestions.append(
                    "💻 Strengthen your technical skills by solving coding problems."
                )

            if communication < 4:
                suggestions.append(
                    "🗣 Improve communication through mock interviews and presentations."
                )

            if internship == "No":
                suggestions.append(
                    "🏢 Complete at least one internship for practical exposure."
                )

            if hackathon == "No":
                suggestions.append(
                    "🏆 Participate in hackathons to improve problem-solving skills."
                )

            if backlogs > 0:
                suggestions.append(
                    "📚 Clear all backlogs before campus placements."
                )

            if major_projects + mini_projects < 2:
                suggestions.append(
                    "🚀 Build more academic or real-world projects."
                )


            if len(suggestions) == 0:

                st.success("""
            🎉 Excellent Profile!

            Your profile looks well-balanced.

            Keep improving your projects and interview preparation.
            """)

            else:

                for item in suggestions:

                    st.warning(item)



            # ==========================================
            # PROFILE STRENGTH
            # ==========================================

            st.markdown("---")

            st.subheader("⭐ Overall Profile Strength")

            score = 0

            score += cgpa

            score += skills

            score += communication * 2

            score += (major_projects + mini_projects)

            score += internship_value * 2

            score += hackathon_value

            score -= backlogs

            if score >= 28:

                st.success("🟢 Excellent Placement Profile")

            elif score >= 22:

                st.info("🟡 Good Placement Profile")

            else:

                st.error("🔴 Needs Significant Improvement")

            # ==========================================
            # PDF REPORT GENERATOR
            # ==========================================

            def generate_pdf():

                buffer = BytesIO()

                doc = SimpleDocTemplate(buffer)

                styles = getSampleStyleSheet()

                story = []

                # ----------------------------
                # Title
                # ----------------------------

                title = Paragraph(
                    "<b><font size=20>AI Student Placement Prediction Report</font></b>",
                    styles["Title"]
                )

                story.append(title)
                story.append(Spacer(1,20))

                # ----------------------------
                # Student Details
                # ----------------------------

                story.append(
                    Paragraph("<b>Student Details</b>",styles["Heading2"])
                )

                story.append(Spacer(1,10))

                data = [

                    ["Feature","Value"],

                    ["CGPA",cgpa],

                    ["Major Projects",major_projects],

                    ["Mini Projects",mini_projects],

                    ["Workshops",workshops],

                    ["Skills Rating",skills],

                    ["Communication",communication],

                    ["Internship",internship],

                    ["Hackathon",hackathon],

                    ["10th Percentage",tenth],

                    ["12th Percentage",twelfth],

                    ["Backlogs",backlogs],

                ]

                table = Table(data)

                table.setStyle(

                    TableStyle([

                        ("BACKGROUND",(0,0),(-1,0),report_colors.darkblue),

                        ("TEXTCOLOR",(0,0),(-1,0),report_colors.white),

                        ("GRID",(0,0),(-1,-1),1,report_colors.black),

                        ("BACKGROUND",(0,1),(-1,-1),report_colors.beige),

                        ("ALIGN",(0,0),(-1,-1),"CENTER"),

                        ("BOTTOMPADDING",(0,0),(-1,0),12),

                    ])

                )

                story.append(table)

                story.append(Spacer(1,25))

                # ----------------------------
                # Prediction
                # ----------------------------

                story.append(
                    Paragraph("<b>Prediction Result</b>",styles["Heading2"])
                )

                story.append(Spacer(1,10))

                status = "PLACED ✅" if prediction==1 else "NOT PLACED ❌"

                story.append(
                    Paragraph(f"<b>Status :</b> {status}",styles["BodyText"])
                )

                story.append(
                    Paragraph(
                        f"<b>Confidence :</b> {confidence:.2f} %",
                        styles["BodyText"]
                    )
                )

                story.append(Spacer(1,20))

                # ----------------------------
                # Suggestions
                # ----------------------------

                story.append(
                    Paragraph("<b>Career Suggestions</b>",styles["Heading2"])
                )

                story.append(Spacer(1,10))

                if len(suggestions)==0:

                    story.append(
                        Paragraph(
                            "Excellent profile. Continue improving interview skills.",
                            styles["BodyText"]
                        )
                    )

                else:

                    for s in suggestions:

                        story.append(
                            Paragraph(f"• {s}",styles["BodyText"])
                        )

                story.append(Spacer(1,20))

                story.append(

                    Paragraph(
                        "<b>Generated by AI Student Placement Predictor</b>",
                        styles["Italic"]
                    )

                )

                doc.build(story)

                buffer.seek(0)

                return buffer



            # ==========================================
            # DOWNLOAD BUTTON
            # ==========================================

            st.markdown("---")

            st.subheader("📄 Download Report")

            pdf = generate_pdf()

            st.download_button(

                label="📥 Download Placement Report",

                data=pdf,

                file_name="Placement_Report.pdf",

                mime="application/pdf",

                use_container_width=True

            )



            # ==========================================
            # THANK YOU SECTION
            # ==========================================

            st.markdown("---")

            st.markdown("""
            <div style="
            background:#1E293B;
            padding:25px;
            border-radius:15px;
            text-align:center;
            ">

            <h2>🎓 Thank You for Using</h2>

            <h1>AI Student Placement Predictor</h1>

            <p>
            This prediction is generated using an Artificial Neural Network (ANN)
            trained on historical placement data.
            </p>

            </div>
            """, unsafe_allow_html=True)


            # ==========================================
            # FOOTER
            # ==========================================

            st.markdown("""

            <hr>

            <div style="text-align:center;color:gray;">

            Developed using ❤️ with

            <b>Python | TensorFlow | Streamlit | Scikit-Learn</b>

            <br><br>

            B.Tech Major Project

            <br>

            2026

            </div>

            """,unsafe_allow_html=True)