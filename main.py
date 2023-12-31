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

close_mean = data['Close'].mean()
close_variance = data['Close'].var()

st.write(f"Trung bình giá đóng cửa: {round(close_mean,2)}")
st.write(f"Phương sai giá đóng cửa: {round(close_variance,2)}")

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

def plot_prediction_price(test_predictions):
    plt.figure(figsize=(12, 6))
    data['Close'].plot(legend=True, label='Actual Close Prices')
    test_predictions.plot(legend=True, label='Predicted Close Prices')
    plt.title('Actual and Predicted Close Prices using Holt-Winters')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    st.pyplot(plt)
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

fitted_model, test_predictions = train_holt_winters(data, 1)
plot_prediction_price(test_predictions)
final_prediction = test_predictions.iloc[-1]
st.write(f"Giá trị dự đoán cho ngày tiếp theo là: {final_prediction}")
#####################################################################################################
def exponential_smoothing(data, n_predictions):
    alpha = 0.1
    data['Exp_Smoothing'] = data['Close'].ewm(alpha=alpha, adjust=False).mean()
    
    for i in range(1, n_predictions + 1):
        last_date = data['Date'].iloc[-1] + pd.DateOffset(days=i)
        last_value = data['Exp_Smoothing'].iloc[-1]
        data = pd.concat([data, pd.DataFrame({'Date': [last_date], 'Exp_Smoothing': [last_value]})], ignore_index=True)

    return data
####################################################################################################
col1, col2 = st.columns(2)

with col1:
    model_options = st.selectbox(
        'Lựa chọn mô hình dự đoán:',
        ('Moving average period 3', 'Moving average period 6','Exponential Smoothing','Holt-Winter') 
    )
    if model_options == 'Holt-Winter' or model_options == 'Exponential Smoothing':
            prediction_day = st.number_input('Ngày dự đoán', min_value=1, step=1, value=1)
    prediction_button = st.button("Dự đoán")

with col2:
    if prediction_button:
        if model_options == 'Holt-Winter':
            fitted_model, test_predictions = train_holt_winters(data, prediction_day)
            plot_prediction_price(test_predictions)
            final_prediction = test_predictions.iloc[-1]
            st.write(f"Giá trị dự đoán cho {prediction_day} ngày tiếp theo là: {round(final_prediction,2)}")
        
        elif model_options == 'Moving average period 3':
            data['3_day_mavg'] = data['Close'].rolling(3).mean()
            st.write(data[['Date','Close','3_day_mavg']].tail())
            fig = px.line(data, x=data.index, y=['Close', '3_day_mavg'], labels={'value': 'Price', 'variable': 'Metric', 'index': 'Date'})
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price',
                title='Moving average with period 3'
            )
            st.plotly_chart(fig)
        
        
        elif model_options == 'Moving average period 6':
            data['6_day_mavg'] = data['Close'].rolling(6).mean()
            st.write(data[['Date','Close','6_day_mavg']].tail())
            fig = px.line(data, x=data.index, y=['Close', '6_day_mavg'], labels={'value': 'Price', 'variable': 'Metric', 'index': 'Date'})
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price',
                title='Moving average with period 6'
            )
            st.plotly_chart(fig)
        
        elif model_options == 'Exponential Smoothing':
            exponential_smoothing(data, prediction_day)
            fig = px.line(data, x='Date', y=['Close', 'Exp_Smoothing'], labels={'value': 'Price', 'variable': 'Metric', 'Date': 'Date'})
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price',
                title=f'Exponential Smoothing with alpha = 0.1'
            )
            st.plotly_chart(fig)
            predicted_price = data['Exp_Smoothing'].iloc[-1]
            st.write(f"Giá dự đoán cho ngày cuối cùng trong {prediction_day} ngày tiếp theo là: {round(predicted_price,2)}")
