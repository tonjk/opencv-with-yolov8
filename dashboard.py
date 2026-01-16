import streamlit as st
import pandas as pd
import sqlite3

st.title("Live Demographics Dashboard")

# Connect to DB
conn = sqlite3.connect('demographics.db')

# Auto-refresh logic could go here, but for now we load static
df = pd.read_sql_query("SELECT * FROM detections ORDER BY timestamp DESC", conn)

if not df.empty:
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Detections", len(df))
    col2.metric("Avg Age", int(df['age'].mean()))
    col3.metric("Most Frequent Gender", df['gender'].mode()[0])

    # Charts
    st.subheader("Age Distribution")
    st.bar_chart(df['age'].value_counts())

    st.subheader("Recent Data Log")
    st.dataframe(df)
else:
    st.warning("No data collected yet. Run the collector script.")