import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Software Risk Predictor",
    page_icon="⚠️",
    layout="centered"
)

with open('scaler.pkl','rb') as f: # 'rb' = read binary
    scaler = pickle.load(f)         # .pkl (pickle) files store data in binary format.

st.title("Software Project Delivery Risk Predictor")
st.write("Enter your software module details below to get a risk score.")

st.subheader("Enter Module Details")

vg = st.slider("Cyclomatic Complexity v(g)", 
               min_value=1, max_value=100, value=10)

loc = st.slider("Lines of Code (loc)", 
                min_value=1, max_value=1000, value=50)

d = st.slider("Halstead Difficulty (d)", 
              min_value=1, max_value=100, value=20)

evg = st.slider("Essential Complexity ev(g)", 
                min_value=1, max_value=50, value=5)

uniq_opnd = st.slider("Unique Operands", 
                      min_value=1, max_value=200, value=30)


# Calculate risk score
input_data = np.log1p([[vg, loc, d, evg, uniq_opnd]])
scaled_input = scaler.transform(input_data)

risk_score = (
    0.25 * scaled_input[:,0] +
    0.25 * scaled_input[:,1] +
    0.20 * scaled_input[:,2] +
    0.20 * scaled_input[:,3] +
    0.10 * scaled_input[:,4]
) * 100

risk_score = round(float(risk_score), 1)

# Show result
st.divider()
st.subheader("Risk Score Result")
st.metric("Risk Score", f"{risk_score} / 100")

# Show risk level
if risk_score <= 33:
    st.success("LOW RISK — Module is probably clean")
elif risk_score <= 66:
    st.warning("MEDIUM RISK — Needs attention")
else:
    st.error("HIGH RISK — Likely to have defects")

