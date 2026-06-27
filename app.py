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
st.write('Aplikasi prediksi churn menggunakan model **Random Forest** terbaik.')

st.sidebar.header('Input Data Pelanggan')

total_spent               = st.sidebar.number_input('Total Pengeluaran',                    min_value=0.0)
satisfaction_score        = st.sidebar.number_input('Skor Kepuasan',                        min_value=0.0, max_value=10.0)
support_tickets           = st.sidebar.number_input('Jumlah Tiket Dukungan',                min_value=0)
avg_session_time          = st.sidebar.number_input('Rata-rata Waktu Sesi',                 min_value=0.0)
pages_per_session         = st.sidebar.number_input('Rata-rata Halaman per Sesi',           min_value=0.0)
marketing_spend_per_user  = st.sidebar.number_input('Biaya Pemasaran per Pelanggan',        min_value=0.0)
avg_order_value           = st.sidebar.number_input('Rata-rata Nilai Transaksi',            min_value=0.0)
lifetime_value            = st.sidebar.number_input('Nilai Total Pelanggan',                min_value=0.0)
email_open_rate           = st.sidebar.number_input('Persentase Email Dibuka',              min_value=0.0, max_value=1.0)
email_click_rate          = st.sidebar.number_input('Persentase Klik Email',                min_value=0.0, max_value=1.0)
age                       = st.sidebar.number_input('Usia',                                 min_value=0)
total_visits              = st.sidebar.number_input('Total Kunjungan',                      min_value=0)
last_3_month_purchase_freq= st.sidebar.number_input('Frekuensi Pembelian 3 Bulan Terakhir',  min_value=0)
nps_score                 = st.sidebar.number_input('Net Promoter Score (NPS)',             min_value=0)
delivery_delay_days       = st.sidebar.number_input('Keterlambatan Pengiriman (hari)',      min_value=0)

if st.sidebar.button('Prediksi'):
    input_data = pd.DataFrame([[
        total_spent, satisfaction_score, support_tickets, avg_session_time,
        pages_per_session, marketing_spend_per_user, avg_order_value, lifetime_value,
        email_open_rate, email_click_rate, age, total_visits,
        last_3_month_purchase_freq, nps_score, delivery_delay_days
    ]], columns=top_features)

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
