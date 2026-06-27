import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model        = joblib.load(os.path.join(BASE_DIR, 'best_model.joblib'))
scaler       = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
top_features = joblib.load(os.path.join(BASE_DIR, 'top_features.joblib'))

st.title('🔍 Prediksi Churn Pelanggan')
st.write('Aplikasi prediksi churn menggunakan model terbaik (sesuai top_features hasil training).')

st.sidebar.header('Input Data Pelanggan')

# ==========================================================
# Mapping encoding kategorikal -- HASIL LabelEncoder (alfabetis)
# Ini HARUS sama dengan urutan alfabetis nilai unik di dataset asli.
# Kalau nanti ada nilai baru di dataset (mis. negara baru), mapping
# ini harus diupdate manual juga.
# ==========================================================
MAP_COUNTRY = {'Bangladesh': 0, 'Germany': 1, 'India': 2, 'UK': 3, 'USA': 4}
MAP_PAYMENT = {'BKash': 0, 'Card': 1, 'PayPal': 2, 'SEPA': 3, 'UPI': 4}
MAP_DEVICE  = {'Desktop': 0, 'Mobile': 1, 'Tablet': 2}
MAP_CITY    = {'Berlin': 0, 'Delhi': 1, 'Dhaka': 2, 'Hamburg': 3,
               'London': 4, 'Mumbai': 5, 'New York': 6}

CATEGORICAL_MAPS = {
    'country': MAP_COUNTRY,
    'payment_method': MAP_PAYMENT,
    'device_type': MAP_DEVICE,
    'city': MAP_CITY,
}

# Urutan top_features (sesuai notebook):
# ['total_spent', 'satisfaction_score', 'support_tickets', 'country',
#  'payment_method', 'device_type', 'city', 'last_3_month_purchase_freq',
#  'marketing_spend_per_user', 'nps_score', 'delivery_delay_days',
#  'lifetime_value', 'avg_session_time', 'avg_order_value', 'email_open_rate']

input_values = {}

for fitur in top_features:
    label = fitur.replace('_', ' ').title()

    if fitur in CATEGORICAL_MAPS:
        pilihan = st.sidebar.selectbox(label, list(CATEGORICAL_MAPS[fitur].keys()))
        input_values[fitur] = CATEGORICAL_MAPS[fitur][pilihan]

    elif fitur == 'satisfaction_score':
        input_values[fitur] = st.sidebar.number_input(label, min_value=0.0, max_value=10.0, step=0.1)

    elif fitur == 'email_open_rate':
        input_values[fitur] = st.sidebar.number_input(label, min_value=0.0, max_value=1.0, step=0.01)

    elif fitur in ['support_tickets', 'last_3_month_purchase_freq', 'nps_score', 'delivery_delay_days']:
        input_values[fitur] = st.sidebar.number_input(label, min_value=0, step=1)

    else:
        # total_spent, marketing_spend_per_user, lifetime_value,
        # avg_session_time, avg_order_value -> numerik bebas (USD/menit)
        input_values[fitur] = st.sidebar.number_input(label, min_value=0.0)

if st.sidebar.button('Prediksi'):
    # Urutan kolom HARUS sama persis dengan top_features
    input_data = pd.DataFrame([[input_values[f] for f in top_features]], columns=top_features)

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0]

    st.subheader('Hasil Prediksi')
    if prediction == 1:
        st.error(f'⚠️ Pelanggan BERPOTENSI CHURN dengan probabilitas {probability[1]:.2%}')
    else:
        st.success(f'✅ Pelanggan TIDAK CHURN dengan probabilitas {probability[0]:.2%}')

    st.subheader('Detail Probabilitas')
    st.write(f'- Tidak Churn : {probability[0]:.2%}')
    st.write(f'- Churn       : {probability[1]:.2%}')

    with st.expander('Lihat data input ke model (debug)'):
        st.write(input_data)
