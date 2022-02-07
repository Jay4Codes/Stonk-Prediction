from matplotlib import markers
from nbformat import write
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pandas_datareader as data
import datetime
import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from streamlit_folium import folium_static
import folium
import streamlit as st

st.set_page_config(
   page_title="Stonks Exchange",
   page_icon="./logo.png",
   layout="wide",
   initial_sidebar_state="expanded",
)

st.sidebar.image("./static/images/my_logo.png")
st.sidebar.title('Stonks Exchange')
rad1 =st.sidebar.radio("Navigation",["Home","Profile", "About-Us"])

if rad1 == "Home": 

    model = load_model('./Future Stock Prediction Using Last 7 Days Moving Averages.h5')

    START = "2010-01-01"
    TODAY = datetime.date.today() + datetime.timedelta(7)

    st.title('Stock Prediction')

    

    stocks = ('AURIONPRO.NS', 'GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 10)
    period = n_years * 365

    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text("Load data....")
    data = load_data(selected_stock)
    data_load_state.text("Loading data....Done!")

    st.subheader('Raw Data')
    st.write(data.tail())

    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text=selected_stock, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    df = data.copy()

    df.reset_index(inplace=True)

    x_future = np.array(df.Close.rolling(7).mean()[-7:])

    scale = max(x_future) - min(x_future)
    minimum = min(x_future)

    for i in range(0, len(x_future)):
        x_future[i] = (x_future[i] - minimum) / scale

    x_future = np.reshape(x_future, (1, 7, 1))

    y_future = []

    while len(y_future) < 7:
    #     Predicting future values using 7-day moving averages of the last day 7 days.
        p = model.predict(x_future)[0]
        
    #     Appending the predicted value to y_future
        y_future.append(p)
        
    #     Updating input variable, x_future
        x_future = np.roll(x_future, -1)
        x_future[-1] = p

    y_future = np.array(y_future)
    y_future = np.reshape(y_future, (7))

    for i in range(0, len(y_future)):
        y_future[i] = (y_future[i] * scale) + minimum

    y_future = np.reshape(y_future, (7, 1))

    last7 = pd.DataFrame(df.Close[-7:])
    last7.reset_index(drop=True, inplace=True)
    y_future = pd.DataFrame(y_future, columns=['Close'])
    predictions = pd.concat([last7, y_future], ignore_index=True)

    prev_7 = datetime.date.today() - datetime.timedelta(7)
    predictions['Date'] = [prev_7 + datetime.timedelta(x) for x in range(0, 14)]

    st.subheader('Predicted Data')
    st.write(predictions[8:13])
    


    def plot_predicted_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'][-7:], y=data['Close'][-7:], name='current_stock_price'))
        
        fig.add_trace(go.Scatter(x=predictions['Date'][8:13], y=predictions['Close'][7:], name='predicted_stock_price'))
        fig.add_trace(go.Scatter(x=predictions['Date'][7:9], y=predictions['Close'][6:8], name = '--', mode='lines', line=dict(color='royalblue', width=4, dash='dot')))
        fig.layout.update(title_text = selected_stock, xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_predicted_data()

if rad1 == "Profile":
    st.title("Your Profile")

    col1 , col2 = st.columns(2)

    rad2 =st.radio("Profile",["Sign-Up","Sign-In"])


    if rad2 == "Sign-Up":

        st.title("Registration Form")



        col1 , col2 = st.columns(2)

        fname = col1.text_input("First Name",value = "first name")

        lname = col2.text_input("Second Name")

        col3 , col4 = st.columns([3,1])

        email = col3.text_input("Email ID")

        phone = col4.text_input("Mob number")

        col5 ,col6 ,col7  = st.columns(3)

        username = col5.text_input("Username")

        password =col6.text_input("Password", type = "password")

        col7.text_input("Repeat Password" , type = "password")

        but1,but2,but3 = st.columns([1,4,1])

        agree  = but1.checkbox("I Agree")

        if but3.button("Submit"):
            if agree:  
                st.subheader("Additional Details")

                address = st.text_area("Tell Us Something About You")
                st.write(address)

                st.date_input("Enter your birth-date")

                v1 = st.radio("Gender",["Male","Female","Others"],index = 1)

                st.write(v1)

                st.slider("age",min_value = 18,max_value=60,value = 30,step = 2)

                img = st.file_uploader("Upload your profile picture")
                if img is not None:
                    st.image(img)

            else:
                st.warning("Please Check the T&C box")

    if rad2 == "Sign-In":
        col1 , col2 = st.columns(2)

        username = col1.text_input("Username")

        password =col2.text_input("Password", type = "password")

        but1,but2,but3 = st.columns([1,4,1])

        agree  = but1.checkbox("I Agree")

        if but3.button("Submit"):
            
            if agree:  
                st.subheader("Additional Details")

                address = st.text_area("Tell Us Something About You")
                st.write(address)

                st.date_input("Enter your birth-date")

                v1 = st.radio("Gender",["Male","Female","Others"],index = 1)

                st.write(v1)

                st.slider("age",min_value = 18,max_value=60,value = 30,step = 2)

                img = st.file_uploader("Upload your profile picture")
                if img is not None:
                    st.image(img)
            else:
                st.warning("Please Check the T&C box")

if rad1 == "About-Us": 
    st.title("Stonks Exchange")

    st.subheader('Locate Us')
    m = folium.Map(location=[18.930131167156954, 72.83363330215157], zoom_start=16)

    # add marker for Bombay Stock Exhcange
    tooltip = "Bombay Stock Exhcange"
    folium.Marker(
        [18.930131167156954, 72.83363330215157], popup="Bombay Stock Exhcange", tooltip=tooltip
    ).add_to(m)

    # call to render Folium map in Streamlit
    folium_static(m)