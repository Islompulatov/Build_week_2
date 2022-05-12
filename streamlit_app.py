import streamlit as st

st.title("Google Fit App")

st.write("""
# Find calories which you need
Which one is the best
""")
target_name = st.sidebar.selectbox('Select Activate',('Still', 'Car', 'Train', 'Bus', 'Walking'))
st.write(target_name)

select_classifier = st.sidebar.selectbox('Select Classifier',('Random Forest', 'KNN','Extra Trees', 'Gradient Boost'))
