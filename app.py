import streamlit as st

pg = st.navigation([st.Page("app_clustering.py"), st.Page("app_process_clustering.py"), st.Page("app_outliers.py"), st.Page("app_process_outliers.py")])
pg.run()