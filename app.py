# =========================
# Streamlit Dashboard: Life Expectancy Analysis
# Fully Portable Version
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------
# 1️⃣ Page Config
# -----------------------------
st.set_page_config(page_title="Life Expectancy Dashboard", layout="wide")
st.title("🌍 Global Life Expectancy Dashboard")

sns.set_style("darkgrid")  # Nice plot style

# -----------------------------
# 2️⃣ Portable CSV Loading
# -----------------------------
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))  # For app.py
except NameError:
    base_dir = os.getcwd()  # For notebook

csv_file = os.path.join(base_dir, 'data', 'LifeExpectancyData.csv')

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    st.success(f"✅ Dataset loaded successfully: {df.shape}")
else:
    st.error(f"❌ CSV file not found at {csv_file}")
    st.stop()

# -----------------------------
# 3️⃣ Clean Columns
# -----------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
df = df.dropna(subset=['life_expectancy'])

# -----------------------------
# 4️⃣ Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
year_options = df['year'].sort_values().unique()
selected_year = st.sidebar.selectbox("Select Year", year_options, index=len(year_options)-1)

status_options = df['status'].unique()
selected_status = st.sidebar.multiselect("Select Country Status", status_options, default=status_options)

if 'continent' in df.columns:
    continent_options = df['continent'].unique()
    selected_continent = st.sidebar.multiselect("Select Continent", continent_options, default=continent_options)
else:
    selected_continent = ['All']

# Apply filters
df_filtered = df[df['year'] == selected_year]
df_filtered = df_filtered[df_filtered['status'].isin(selected_status)]
if 'continent' in df.columns:
    df_filtered = df_filtered[df_filtered['continent'].isin(selected_continent)]

st.markdown(f"### Dataset for Year {selected_year}")
st.dataframe(df_filtered.head())

# -----------------------------
# 5️⃣ Analysis 1: Top 10 Countries by Life Expectancy
# -----------------------------
st.subheader("Top 10 Countries by Life Expectancy")
top10 = df_filtered.nlargest(10, 'life_expectancy')
plt.figure(figsize=(10,5))
sns.barplot(x='life_expectancy', y='country', data=top10, color='teal')  # avoid FutureWarning
plt.xlabel("Life Expectancy (Years)")
plt.ylabel("Country")
plt.title(f"Top 10 Countries in {selected_year}")
st.pyplot(plt)

# -----------------------------
# 6️⃣ Analysis 2: Life Expectancy vs GDP
# -----------------------------
st.subheader("Life Expectancy vs GDP per Capita")
df_gdp = df_filtered.dropna(subset=['gdp', 'life_expectancy'])
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_gdp, x='gdp', y='life_expectancy', hue='status', palette='Set2', s=100, alpha=0.7)
plt.xscale('log')
plt.xlabel("GDP per Capita (US$)")
plt.ylabel("Life Expectancy (Years)")
plt.title(f"GDP vs Life Expectancy in {selected_year}")
plt.legend(title='Status')
st.pyplot(plt)

# -----------------------------
# 7️⃣ Analysis 3: Adult Mortality vs Life Expectancy
# -----------------------------
st.subheader("Adult Mortality vs Life Expectancy")
df_adult = df_filtered.dropna(subset=['adult_mortality', 'life_expectancy'])
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_adult, x='adult_mortality', y='life_expectancy', hue='status',
                palette='plasma', s=80, alpha=0.7)
plt.xlabel("Adult Mortality (per 1000 adults)")
plt.ylabel("Life Expectancy (Years)")
plt.title(f"Adult Mortality vs Life Expectancy in {selected_year}")
st.pyplot(plt)

# -----------------------------
# 8️⃣ Analysis 4: Health Expenditure vs Life Expectancy (Bubble Plot)
# -----------------------------
st.subheader("Health Expenditure vs Life Expectancy (Bubble size = GDP)")
df_health = df_filtered.dropna(subset=['percentage_expenditure', 'life_expectancy', 'gdp'])
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_health, x='percentage_expenditure', y='life_expectancy',
                size='gdp', hue='status', palette='viridis', sizes=(50,500), alpha=0.7)
plt.xlabel("Health Expenditure (% of GDP)")
plt.ylabel("Life Expectancy (Years)")
plt.title(f"Health Expenditure vs Life Expectancy in {selected_year}")
st.pyplot(plt)

# -----------------------------
# 9️⃣ Analysis 5: BMI vs Life Expectancy (Regression)
# -----------------------------
st.subheader("BMI vs Life Expectancy")
df_bmi = df_filtered.dropna(subset=['bmi','life_expectancy'])
plt.figure(figsize=(10,6))
sns.regplot(data=df_bmi, x='bmi', y='life_expectancy',
            scatter_kws={'alpha':0.5, 'color':'orange'},
            line_kws={'color':'blue'})
plt.xlabel("Average BMI")
plt.ylabel("Life Expectancy (Years)")
plt.title(f"BMI vs Life Expectancy in {selected_year}")
st.pyplot(plt)

# -----------------------------
# 10️⃣ Analysis 6: HIV/AIDS Prevalence vs Life Expectancy (Log scale)
# -----------------------------
st.subheader("HIV/AIDS Prevalence vs Life Expectancy (Log Scale)")
df_hiv = df_filtered.dropna(subset=['hiv/aids','life_expectancy'])
plt.figure(figsize=(10,6))
sns.scatterplot(data=df_hiv, x='hiv/aids', y='life_expectancy', hue='status',
                palette='Reds', s=80, alpha=0.7)
plt.xscale('log')
plt.xlabel("HIV/AIDS Prevalence (Log Scale)")
plt.ylabel("Life Expectancy (Years)")
plt.title(f"HIV/AIDS vs Life Expectancy in {selected_year}")
st.pyplot(plt)

# -----------------------------
# 11️⃣ Analysis 7: Schooling vs Life Expectancy (Regression)
# -----------------------------
st.subheader("Life Expectancy vs Average Years of Schooling")
df_school = df_filtered.dropna(subset=['schooling','life_expectancy'])
plt.figure(figsize=(10,6))
sns.regplot(data=df_school, x='schooling', y='life_expectancy',
            scatter_kws={'alpha':0.5, 'color':'green'},
            line_kws={'color':'red'})
plt.xlabel("Average Years of Schooling")
plt.ylabel("Life Expectancy (Years)")
plt.title(f"Schooling vs Life Expectancy in {selected_year}")
st.pyplot(plt)
