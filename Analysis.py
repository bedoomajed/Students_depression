
import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import gaussian_kde
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

import sys




st.set_page_config(page_title="Explatory Data Analysis", page_icon="📈", layout="wide")

# Custom heading with dark blue background and white text
def heading():
    st.markdown("""  
        <style>
        .custom-heading {
            background-color: #F5F5DC;  /* اللون البيج */
            color: #333333;             /* خط غامق عشان يبان */
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
            📈 Descriptive Analytics 📊
        </div>
    """, unsafe_allow_html=True)




#remove default theme
theme_plotly = None # None or streamlit

 
st.markdown("""
    <style>
    /* Change st.info background to beige */
    div.stAlert {
        background-color: #F5DEB3 !important;   /* Beige */
        border-left: 6px solid #C5A880 !important; /* Dark beige border */
    }.plot-container > div {
    box-shadow: 0 0 4px #cccccc;
    padding: 10px;
    }

    /* Change text color inside st.info */
    div.stAlert p {
        color: #4a3f35 !important;  /* Dark brown text */
        font-weight: bold;
    }
    
    </style>
""", unsafe_allow_html=True)





# Reading parts
@st.cache_data
def load_data():
    df = pd.read_parquet(r"https://raw.githubusercontent.com/MohamedHeshamrg/Students_depression/main/data/preprocessed/data.parquet")
    return df
df = load_data()



import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

st.title("📊 Mental Health Analysis Dashboard")

st.write("### Comprehensive Analysis of Depression & Related Factors")

# ==========================
# 1. MULTI-PLOTS DASHBOARD
# ==========================

fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle('Mental Health Factors Analysis Dashboard', fontsize=20, fontweight='bold', y=1.02)

# 1.1 Depression Distribution
depression_counts = df['Depression'].value_counts()
axes[0, 0].pie(depression_counts.values, labels=['No Depression', 'Depression'], 
               autopct='%1.1f%%', colors = ['#3EB489', '#66b3ff'])  
axes[0, 0].set_title('Depression Distribution', fontsize=14, fontweight='bold')

# 1.2 Suicidal Thoughts vs Depression
suicidal_depression = pd.crosstab(df['Have you ever had suicidal thoughts ?'], df['Depression'])
suicidal_depression.plot(kind='bar', ax=axes[0, 1], color=['#3EB489', '#66b3ff'])
axes[0, 1].set_title('Suicidal Thoughts vs Depression')
axes[0, 1].legend(['No Depression', 'Depression'])

# 1.3 Sleep Duration vs Depression
sleep_order = ["'5-6 hours'", "'Less than 5 hours'", "'7-8 hours'", "'More than 8 hours'"]
sleep_depression = pd.crosstab(df['Sleep Duration'], df['Depression']).reindex(sleep_order)
sleep_depression.plot(kind='bar', ax=axes[0, 2], color=['#3EB489', '#66b3ff'])
axes[0, 2].set_title('Sleep Duration vs Depression', fontsize=14, fontweight='bold')

# 1.4 Academic Pressure vs Study Satisfaction
pressure_satisfaction = df.groupby('Academic Pressure')['Study Satisfaction'].mean()
axes[1, 0].bar(pressure_satisfaction.index, pressure_satisfaction.values, color='#3EB489')
axes[1, 0].set_title('Academic Pressure vs Study Satisfaction')

# 1.5 Dietary Habits
diet_counts = df['Dietary Habits'].value_counts()
axes[1, 1].bar(diet_counts.index, diet_counts.values, color=['#3EB489', '#c44e52', '#8172b2'])
axes[1, 1].set_title('Dietary Habits Distribution')

# 1.6 Age Distribution
axes[1, 2].hist(df['Age'], bins=20, color='#4c72b0', edgecolor='black')
axes[1, 2].set_title('Age Distribution')

# 1.7 Financial Stress vs Depression Rate
financial_depression = df.groupby('Financial Stress')['Depression'].mean() * 100
axes[2, 0].plot(financial_depression.index, financial_depression.values, marker='o', color='#3EB489')
axes[2, 0].set_title('Financial Stress vs Depression Rate (%)')

# 1.8 Study Hours Dist.
axes[2, 1].hist(df['Work/Study Hours'], bins=20, color='#3EB489', edgecolor='black')
axes[2, 1].set_title('Work/Study Hours Distribution')

# 1.9 Family Mental Illness
family_counts = df['Family History of Mental Illness'].value_counts()
axes[2, 2].pie(family_counts.values, labels=family_counts.index, autopct='%1.1f%%',
               colors=['#3EB489', '#66b3ff'])
axes[2, 2].set_title('Family Mental Illness History')

st.pyplot(fig)

# ==========================
# 2. CORRELATION HEATMAP
# ==========================

st.write("## 🔥 Correlation Heatmap")

fig2, ax = plt.subplots(figsize=(12, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title('Correlation Matrix of Numerical Variables')

st.pyplot(fig2)

# ==========================
# 3. DETAILED ANALYSIS SET
# ==========================

st.write("## 🔍 Detailed Factor Analysis")

fig3, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.boxplot(data=df, x='Depression', y='GPA', ax=axes[0, 0], palette=['#3EB489', '#66b3ff'])
axes[0, 0].set_title('GPA by Depression')

sns.scatterplot(data=df, x='Academic Pressure', y='Study Satisfaction', hue='Depression', ax=axes[0, 1])
axes[0, 1].set_title('Pressure vs Satisfaction')

sleep_numeric = pd.Categorical(df['Sleep Duration'],
                               categories=['Less than 5 hours','5-6 hours','7-8 hours','More than 8 hours'],
                               ordered=True)
df['Sleep_Numeric'] = sleep_numeric.codes

sns.violinplot(data=df, x='Sleep Duration', y='Work/Study Hours',
               hue='Depression', split=True, ax=axes[1, 0])
axes[1, 0].set_title('Sleep vs Work Hours')

sns.kdeplot(data=df[df['Gender']=='Male'], x='Age', hue='Depression', ax=axes[1, 1], fill=True)
sns.kdeplot(data=df[df['Gender']=='Female'], x='Age', hue='Depression', ax=axes[1, 1], fill=True)
axes[1, 1].set_title('Age Dist by Gender & Depression')

st.pyplot(fig3)

# ==========================
# 4. CITY-WISE ANALYSIS
# ==========================

st.write("## 🏙️ City-wise Mental Health Overview")

fig4, axes = plt.subplots(1, 2, figsize=(18, 8))

city_counts = df['City'].value_counts().head(10)
axes[0].barh(city_counts.index, city_counts.values)
axes[0].set_title('Top 10 Cities by Sample Size')

city_depression_rate = df[df['City'].isin(city_counts.index)].groupby('City')['Depression'].mean()*100
axes[1].barh(city_depression_rate.index, city_depression_rate.values)
axes[1].set_title('Depression Rate in Top 10 Cities (%)')

st.pyplot(fig4)

# ==========================
# 5. DEGREE ANALYSIS
# ==========================

st.write("## 🎓 Degree-wise Mental Health Analysis")

fig5, axes = plt.subplots(1, 3, figsize=(18, 6))

degree_counts = df['Degree'].value_counts()
axes[0].pie(degree_counts.values, labels=degree_counts.index, autopct='%1.1f%%')
axes[0].set_title('Degree Distribution')

degree_depression = df.groupby('Degree')['Depression'].mean()*100
axes[1].bar(degree_depression.index, degree_depression.values)
axes[1].set_title('Depression Rate by Degree')

degree_gpa = df.groupby('Degree')['GPA'].mean()
axes[2].bar(degree_gpa.index, degree_gpa.values)
axes[2].set_title('Average GPA by Degree')

st.pyplot(fig5)

# ==========================
# SUMMARY
# ==========================

st.write("## 📌 Summary of Key Insights")

st.success(f"**Depression Rate:** {df['Depression'].mean()*100:.1f}%")
st.info(f"**Suicidal Thoughts:** {df['Have you ever had suicidal thoughts ?'].value_counts(normalize=True)['Yes']*100:.1f}%")
st.info(f"**Average Age:** {df['Age'].mean():.1f} years")
st.info(f"**Average GPA:** {df['GPA'].mean():.2f}")
st.info(f"**Most Common Sleep Duration:** {df['Sleep Duration'].mode()[0]}")
st.info(f"**Most Common Dietary Habit:** {df['Dietary Habits'].mode()[0]}")








































                
st.write("📌 **Statistics for Categorical Columns**")
st.dataframe(df.describe(include="O").T)



# ------------------------------







