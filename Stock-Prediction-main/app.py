
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib widget
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime
# with open('styles.css') as f:
#   st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

today=datetime.date.today()
now=datetime.datetime.now()
st.write(today.strftime("%d/%m/%Y"))
st.write(now.strftime("%H:%M:%S"))
st.title("Stock Price Prediction")
user_input = st.selectbox('Enter Stock Ticker',('AAPL','TSLA','MSFT','GOOG','GOOGL','AMZN','HDB','WIT','INFY','MMYT','AZRE','TTM' ))

start = st.date_input('Enter Start Date (YYYY-MM-DD)',datetime.date(2010,1,1))
end = st.date_input('Enter End Date (YYYY-MM-DD)',datetime.date(2019,12,31))
df = data.DataReader(user_input,data_source='yahoo', start=start, end=end)
columns = st.columns((1,1))



# Describing Data
st.subheader('Data from '+str(start.year)+' and '+str(end.year))
st.write(df.describe())

# Visualisations
st.subheader("Closing Price vs Time chart with 100MA & 200MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label='Moving Average for 100 days')
plt.plot(ma200,'g',label='Moving Average for 200 days')
plt.plot(df.Close,'b',label='Close Price')
plt.xlabel('Years')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)
# plt.show()


# Data split
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(data_train.shape)
print(data_test.shape)

# Scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_train)

# Splitting scaled data
# x_train = []
# y_train = []
#
# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])
# x_train,y_train = np.array(x_train),np.array(y_train)

# Load model
model = load_model('keras_model.h5')

# Testing
past_100_days = data_train.tail(100)
final_df = past_100_days.append(data_test,ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test),np.array(y_test)

# Making Predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# Prediction Viz
st.subheader("Predictions vs Actual")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# Accuracy
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse = mean_squared_error(y_test,y_predicted)
r2 = r2_score(y_test,y_predicted)
mae = mean_absolute_error(y_test,y_predicted)
st.subheader("R\u00b2 Score : "+str(r2))
st.subheader("Mean square error : "+str(mse))
st.subheader("Mean absolute error : "+str(mae))


# Regression Model
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# df = data.DataReader(user_input,data_source='yahoo', start=start, end=end)
# df = df.drop(['Date'])
# X = np.array(df.index).reshape(-1,1)
# Y = df['Close']
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=100)
# scaler = StandardScaler().fit(X_train)
# from sklearn.linear_model import LinearRegression
# lm=LinearRegression()
# lm.fit(X_train,Y_train)

# # Reg Viz
# fig3=plt.figure(figsize=(12,6))
# plt.scatter(X_train,Y_train,c='red',label='Actual')
# plt.plot(X_train,lm.predict(X_train),c='black',label='Predicted')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Price')
# st.pyplot(fig3)
