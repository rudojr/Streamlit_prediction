import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from pickle import TRUE


# TODAY = date.today().strftime("%Y-%m-%d")
# START =  TODAY - timedelta(days=365*5)
# START = START.strftime("%Y-%m-%d")

TODAY = datetime.now()
START = TODAY - timedelta(days=365*5)

st.title("STOCK PREDICTION APPLICATION")

stocks = ("AAPL","GOOG","MSFT","GME","TSLA")
selected_stock = st.selectbox("Select dataset from yfinance for prediction", stocks)

# n_years = st.slider("Year of prediction:",1,5)
period = 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data.... done!!")


st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name = 'stock_close'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name = 'stock_high'))
    fig.update_layout(title_text = "Time series data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

max_price = data['High'].max()
min_price = data['Low'].min()

ngay_max_price = pd.Timestamp(data[data['High'] == max_price]['Date'].values[0]).strftime("%Y-%m-%d")
ngay_min_price = pd.Timestamp(data[data['Low'] == min_price]['Date'].values[0]).strftime("%Y-%m-%d")


st.write(f"Cổ phiếu đạt giá trị cao nhất vào ngày {ngay_max_price}, giá trị lúc đó là {int(max_price)}")
st.write(f"Cổ phiếu đạt giá trị thấp nhất vào ngày {ngay_min_price}, giá trị lúc đó là {int(min_price)}")

##########################################################################################################

data['Year'] = pd.to_datetime(data['Date']).dt.year

# Tạo DataFrame chứa giá trị trung bình của các năm
average_prices = data.groupby('Year')['Close'].mean().reset_index()

average_prices_2023 = average_prices[average_prices['Year'] == 2023]
other_years = average_prices[average_prices['Year'] != 2023]
st.subheader('Biểu đồ giá trị trung bình cổ phiếu năm 2023 so với các năm còn lại')
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(other_years['Year'], other_years['Close'], label='Các năm còn lại')
ax.scatter(average_prices_2023['Year'], average_prices_2023['Close'], color='red', label='Năm 2023')
ax.set_xlabel('Năm')
ax.set_ylabel('Giá trị trung bình cổ phiếu')
ax.set_title('Giá trị trung bình cổ phiếu năm 2023 so với các năm còn lại')
ax.legend()
ax.grid(True)

average_price_2023 = average_prices_2023['Close'].values[0]
average_price_other_years = other_years['Close'].mean()

if average_price_2023 > average_price_other_years:
    st.write("Giá trị trung bình cổ phiếu năm 2023 cao hơn so với các năm còn lại.")
elif average_price_2023 < average_price_other_years:
    st.write("Giá trị trung bình cổ phiếu năm 2023 thấp hơn so với các năm còn lại.")
else:
    st.write("Giá trị trung bình cổ phiếu năm 2023 không khác biệt so với các năm còn lại.")

# Hiển thị biểu đồ trên Streamlit
st.pyplot(fig)
###################################################################################################
##Rolling 3
data['3_day_mavg'] = data['Close'].rolling(3).mean()

st.subheader('Dữ liệu cuối cùng')
st.write(data.tail())
fig = px.line(data, x=data.index, y=['Close', '3_day_mavg'], labels={'value': 'Price', 'variable': 'Metric', 'index': 'Date'})
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    title='Moving average with period 3'
)
st.plotly_chart(fig)

##Rolling 6
data['6_day_mavg'] = data['Close'].rolling(6).mean()

st.subheader('Dữ liệu đầu tiên')
st.write(data.head())

st.subheader('Dữ liệu cuối cùng')
st.write(data.tail())
fig = px.line(data, x=data.index, y=['Close', '6_day_mavg'], labels={'value': 'Price', 'variable': 'Metric', 'index': 'Date'})
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    title='Moving average with period 6'
)
st.plotly_chart(fig)


##Alpha 0.1
data['Exp_Smoothing_alpha_0.1'] = data['Close'].ewm(alpha=0.1).mean()

st.subheader('Dữ liệu cuối cùng')
st.write(data.tail())

fig = px.line(data, x=data.index, y=['Close', 'Exp_Smoothing_alpha_0.1'])
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    title='Exponential Smoothing with alpha=0.1'
)
st.plotly_chart(fig)
st.write(f"Predicted close value for {data.index[-1]}: {data.iloc[-1]['Exp_Smoothing_alpha_0.1']}")

#####################################################################################################
option = st.selectbox(
    'Lựa chọn dự đoán:',
    ('Dự báo giá 1 ngày tiếp theo','Dự báo giá 1 tuần tiếp theo','Dự báo giá 1 tháng tiếp theo','Dự báo giá 1 năm tiếp theo') 
)

#Du doan gia với holt winters
def train_holt_winters(data, n_years):
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    train_data = data['Close'].dropna()
    model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=60)
    model_fit = model.fit()
    n_predictions = n_years
    test_predictions = model_fit.forecast(n_predictions)
    test_predictions.index = np.arange(max(data.index), max(data.index) + n_predictions)
    return model_fit, test_predictions

def plot_prediction_price(test_predictions):
    plt.figure(figsize=(12, 6))
    data['Close'].plot(legend=True, label='Actual Close Prices')
    test_predictions.plot(legend=True, label='Predicted Close Prices')
    plt.title('Actual and Predicted Close Prices using Holt-Winters')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    st.pyplot(plt)

if option == 'Dự báo giá 1 ngày tiếp theo':
    fitted_model, test_predictions = train_holt_winters(data, 1)
    plot_prediction_price(test_predictions)
    final_prediction = test_predictions.iloc[-1]
    st.write(f"Giá trị dự đoán cho ngày tiếp theo là: {final_prediction}")
    
elif option == 'Dự báo giá 1 tuần tiếp theo':
    fitted_model, test_predictions = train_holt_winters(data, 7)
    for i in test_predictions:
        st.write(i)
    plot_prediction_price(test_predictions)
    final_prediction = test_predictions.iloc[-1]
    st.write(f"Giá trị dự đoán cho tuần tiếp theo là: {final_prediction}")
    
elif option == 'Dự báo giá 1 tháng tiếp theo':
    fitted_model, test_predictions = train_holt_winters(data, 30)
    st.write("Thông tin tổng quát về dữ liệu dự đoán:")
    st.write(test_predictions.describe())
    plot_prediction_price(test_predictions)
    final_prediction = test_predictions.iloc[-1]
    st.write(f"Giá trị dự đoán cho tháng tiếp theo là: {final_prediction}")
    
elif option == 'Dự báo giá 1 năm tiếp theo':
    fitted_model, test_predictions = train_holt_winters(data, 365)
    st.write("Thông tin tổng quát về dữ liệu dự đoán:")
    st.write(test_predictions.describe())
    plot_prediction_price(test_predictions)
    final_prediction = test_predictions.iloc[-1]
    st.write(f"Giá trị dự đoán cho năm tiếp theo là: {final_prediction}")