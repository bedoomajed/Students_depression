import streamlit as st
import pandas as pd 
import plotly.express as px
import time
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import joblib
from scipy.stats import gaussian_kde
import sys

import joblib
import pandas as pd
import numpy as np

import sys




st.set_page_config(page_title="Students Depression Dashboard", page_icon="🌎", layout="wide")


# Custom heading with beige background and dark text
def heading():
    st.markdown("""  
        <style>
        .custom-heading {
            background-color: #CCF5D3;  /* Mint Green */
            color: #145A32;             /* Dark Green */
            padding: 20px;
            border-radius: 12px;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
            margin-bottom: 25px;
        }
        </style>

        <div class="custom-heading">
            📈 Students Depression Dashboard 📊
        </div>
    """, unsafe_allow_html=True)






#remove default theme
theme_plotly = None # None or streamlit

 
st.markdown("""
    <style>
    /* Change st.info background to mint green */
    div.stAlert {
        background-color: #CCF5D3 !important;   /* Mint Green */
        border-left: 6px solid #27AE60 !important; /* Darker green border */
    }
    
    .plot-container > div {
        box-shadow: 0 0 4px #cccccc;
        padding: 10px;
    }

    /* Change text color inside st.info */
    div.stAlert p {
        color: #145A32 !important;  /* Dark green text */
        font-weight: bold;
    }
    
    </style>
""", unsafe_allow_html=True)


# Reading parts
# Reading parts
@st.cache_data
def load_data():
    df = pd.read_parquet("https://raw.githubusercontent.com/MohamedHeshamrg/Students_depression/main/data/preprocessed/data.parquet")
    return df
df = load_data()








def HomePage():
 heading()
  #1. print dataframe
 with st.expander("🧭 My database"):
  #st.dataframe(df_selection,use_container_width=True)
  st.dataframe(df,use_container_width=True)
# =========================
# 2. Compute Top Analytics
# =========================

 Total_depression = 21          # عدد الطلاب المصابين بالاكتئاب
 Suicidal_Thoughts = 63         # نسبة الذين لديهم أفكار انتحارية
 Sleep_Duration = 30            # نسبة النوم غير الكافي
 The_most_type = 60             # أعلى نسبة في Degree (مثلاً Bachelor)

# =========================
# 3. Columns UI
# =========================

 total1, total2, total3, total4 = st.columns(4, gap='large')

 with total1:
     st.info('Total Depressed Students', icon="🧠")
     st.metric(label='Count', value=f"{Total_depression}")

 with total2:
     st.info('Suicidal Thoughts', icon="⚠️")
     st.metric(label='Percentage', value=f"{Suicidal_Thoughts}%")

 with total3:
     st.info('Poor Sleep Duration', icon="😴")
     st.metric(label='Percentage', value=f"{Sleep_Duration}%")

 with total4:
     st.info('Most Common Degree', icon="🎓")
     st.metric(label='Bachelor', value=f"{The_most_type}%")

 st.markdown("""---""")


 #graphs
 
def Graphs():

    st.markdown("### 📊 Depression Data Visual Analytics")

    # ======================
    # ROW 1 → Histogram + Pie
    # ======================
    with st.container():
        col1, col2 = st.columns([4, 3])

        # Histogram: Age Distribution
        fig = px.histogram(
            df,
            x="Age",
            nbins=20,
            histnorm='density',
            template="plotly_white"
        )

        fig.update_layout(
            title="Age Distribution",
            xaxis_title="Age",
            yaxis_title="Density"
        )

        fig.update_traces(marker=dict(color=px.colors.sequential.Teal[2]))
        col1.plotly_chart(fig, use_container_width=True)

        # Pie Chart: Depression vs Not Depressed
        counts = df["Depression"].value_counts()    # 1 = Depressed, 0 = Not Depressed

        fig = px.pie(
            names=["Not Depressed", "Depressed"],
            values=counts.values,
            title="Depression Rate",
            hole=0.25
        )

        fig.update_traces(marker=dict(colors=px.colors.sequential.Teal))
        fig.update_traces(textposition='inside', textinfo='percent+label')

        col2.plotly_chart(fig, use_container_width=True)

    # ======================
    # ROW 2 → Bar + Violin
    # ======================
    with st.container():
        col1, col2 = st.columns(2)

        # Bar Chart: Depression by Gender
        dep_by_gender = df.groupby("Gender")["Depression"].mean().reset_index()
        dep_by_gender["Depression %"] = dep_by_gender["Depression"] * 100

        fig = px.bar(
            dep_by_gender,
            x="Gender",
            y="Depression %",
            title="Depression Percentage by Gender",
            color="Depression %",
            color_continuous_scale=px.colors.sequential.Teal_r
        )

        col1.plotly_chart(fig, use_container_width=True)

        # Violin Plot: GPA vs Depression
        fig = px.violin(
            df,
            y="GPA",
            x="Depression",
            color="Depression",
            template="plotly_white",
            title="GPA Distribution by Depression Status"
        )

        fig.update_xaxes(
            tickvals=[0, 1],
            ticktext=["Not Depressed", "Depressed"]
        )

        col2.plotly_chart(fig, use_container_width=True)




HomePage()
Graphs()
     
  


















