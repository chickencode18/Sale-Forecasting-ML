import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load trained model
model = joblib.load('model.pkl')

st.title("Sales Forecasting App")

# Nhập ngày cần dự đoán
input_date = st.date_input('Chọn ngày cần dự đoán:', pd.Timestamp.today() + pd.Timedelta(days=1))

# Giả lập dữ liệu rolling (có thể dùng từ cơ sở dữ liệu thực tế)
rolling_sales = [1000.0] * 14  # Đây là doanh thu 14 ngày gần nhất (có thể cập nhật bằng dữ liệu thực tế)

if st.button("Dự đoán doanh thu cho ngày đã chọn"):
    next_date = pd.Timestamp(input_date)

    # Tính toán các đặc trưng
    rolling_mean_7 = np.mean(rolling_sales[-7:])
    rolling_mean_14 = np.mean(rolling_sales[-14:])
    rolling_std_7 = np.std(rolling_sales[-7:])
    growth_rate = (rolling_sales[-1] / rolling_sales[-2] - 1) if rolling_sales[-2] != 0 else 0

    next_day_features = pd.DataFrame({
        'Day_of_Week_sin': [np.sin(2 * np.pi * next_date.dayofweek / 7)],
        'Day_of_Week_cos': [np.cos(2 * np.pi * next_date.dayofweek / 7)],
        'Day_of_Month': [next_date.day],
        'Week_of_Year': [next_date.isocalendar().week],
        'Prev_Day_Sales': [rolling_sales[-1]],
        'Rolling_Mean_7': [rolling_mean_7],
        'Rolling_Mean_14': [rolling_mean_14],
        'Rolling_Std_7': [rolling_std_7],
        'Growth_Rate': [growth_rate]
    })

    prediction = model.predict(next_day_features)[0]
    st.success(f'Dự báo doanh thu cho ngày {next_date.date()}: {prediction:.2f}')
