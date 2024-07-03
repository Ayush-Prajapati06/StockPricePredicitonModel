import numpy as np
import matplotlib.pyplot as plt
import streamlit as st  
import pandas as pd  
import yfinance as yf
import requests
import datetime  
import matplotlib.pyplot as plt  
import time  
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from matplotlib.pyplot import axis
from textblob import TextBlob
from datetime import date
from plotly import graph_objs as go 
from plotly.subplots import make_subplots
from prophet import Prophet  
from prophet.plot import plot_plotly
from streamlit_option_menu import option_menu  

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

def add_meta_tag():
    meta_tag = """
        <head>
            <meta name="google-site-verification" content="QBiAoAo1GAkCBe1QoWq-dQ1RjtPHeFPyzkqJqsrqW-s" />
        </head>
    """
    st.markdown(meta_tag, unsafe_allow_html=True)

# Main code
add_meta_tag()

# Sidebar Section Starts Here
today = date.today()  # today's date
# st.write('''# Stock Project''')  # title
# st.sidebar.image("Images\download-removebg-preview.png", width=250, use_column_width=False)  # logo
st.sidebar.write('''# Stock Analysis''')

with st.sidebar: 
        selected = option_menu("Utilities", ["Stocks Performance Comparison", "Real-Time Stock Price", "Stock Prediction","LSTM","Sentiment Analysis",])

start = st.sidebar.date_input(
    'Start', datetime.date(2022, 1, 1))  
end = st.sidebar.date_input('End', datetime.date.today())  
# Sidebar Section Ends Here

# read csv file
stock_df = pd.read_csv("StockStreamTickersData.csv")

# Stock Performance Comparison Section Starts Here
if(selected == 'Stocks Performance Comparison'):  
    st.subheader("Stocks Performance Comparison")
    tickers = stock_df["Company Name"]
    # dropdown for selecting assets
    dropdown = st.multiselect('Pick your assets', tickers)

    with st.spinner('Loading...'):  
        time.sleep(2)
        # st.success('Loaded')

    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    symb_list = []  
    for i in dropdown:  
        val = dict_csv.get(i)  
        symb_list.append(val) 

    def relativeret(df):  
        rel = df.pct_change()  
        cumret = (1+rel).cumprod() - 1  
        cumret = cumret.fillna(0) 
        return cumret  

    if len(dropdown) > 0:  # if user selects atleast one asset
        df = relativeret(yf.download(symb_list, start, end))[
            'Adj Close']  # download data from yfinance
        # download data from yfinance
        raw_df = relativeret(yf.download(symb_list, start, end))
        raw_df.reset_index(inplace=True)  # reset index

        closingPrice = yf.download(symb_list, start, end)[
            'Adj Close']  # download data from yfinance
        volume = yf.download(symb_list, start, end)['Volume']
        
        st.subheader('Raw Data {}'.format(dropdown))
        st.write(raw_df)  # display raw data
        chart = ('Line Chart', 'Area Chart', 'Bar Chart')  # chart types
        # dropdown for selecting chart type
        dropdown1 = st.selectbox('Pick your chart', chart)
        with st.spinner('Loading...'):  # spinner while loading
            time.sleep(2)

        st.subheader('Relative Returns {}'.format(dropdown))
                
        if (dropdown1) == 'Line Chart':  # if user selects 'Line Chart'
            st.line_chart(df)  # display line chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  # display line chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume)  # display line chart

        elif (dropdown1) == 'Area Chart':  # if user selects 'Area Chart'
            st.area_chart(df)  # display area chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.area_chart(closingPrice)  # display area chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.area_chart(volume)  # display area chart

        elif (dropdown1) == 'Bar Chart':  # if user selects 'Bar Chart'
            st.bar_chart(df)  # display bar chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.bar_chart(closingPrice)  # display bar chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.bar_chart(volume)  # display bar chart

        else:
            st.line_chart(df, width=1000, height=800,
                          use_container_width=False)  # display line chart
            # display closing price of selected assets
            st.write("### Closing Price of {}".format(dropdown))
            st.line_chart(closingPrice)  # display line chart

            # display volume of selected assets
            st.write("### Volume of {}".format(dropdown))
            st.line_chart(volume)  # display line chart

    else:  # if user doesn't select any asset
        st.write('Please select atleast one asset')  # display message
# Stock Performance Comparison Section Ends Here
    
# Real-Time Stock Price Section Starts Here
elif(selected == 'Real-Time Stock Price'):  # if user selects 'Real-Time Stock Price'
    st.subheader("Real-Time Stock Price")
    tickers = stock_df["Company Name"]  # get company names from csv file
    # dropdown for selecting company
    a = st.selectbox('Pick a Company', tickers)

    with st.spinner('Loading...'):  # spinner while loading
            time.sleep(2)

    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    symb_list = []  # list for storing symbols

    val = dict_csv.get(a)  # get symbol from csv file
    symb_list.append(val)  # append symbol to list

    if "button_clicked" not in st.session_state:  # if button is not clicked
        st.session_state.button_clicked = False  # set button clicked to false

    def callback():  # function for updating data
        # if button is clicked
        st.session_state.button_clicked = True  # set button clicked to true
    if (
        st.button("Search", on_click=callback)  # button for searching data
        or st.session_state.button_clicked  # if button is clicked
    ):
        if(a == ""):  # if user doesn't select any company
            st.write("Click Search to Search for a Company")
            with st.spinner('Loading...'):  # spinner while loading
             time.sleep(2)
        else:  # if user selects a company
            # download data from yfinance
            data = yf.download(symb_list, start=start, end=end)
            data.reset_index(inplace=True)  # reset index
            st.subheader('Raw Data of {}'.format(a))  # display raw data
            st.write(data)  # display data

            def plot_raw_data():  # function for plotting raw data
                fig = go.Figure()  # create figure
                fig.add_trace(go.Scatter(  # add scatter plot
                    x=data['Date'], y=data['Open'], name="stock_open"))  # x-axis: date, y-axis: open
                fig.add_trace(go.Scatter(  # add scatter plot
                    x=data['Date'], y=data['Close'], name="stock_close"))  # x-axis: date, y-axis: close
                fig.layout.update(  # update layout
                    title_text='Line Chart of {}'.format(a) , xaxis_rangeslider_visible=True)  # title, x-axis: rangeslider
                st.plotly_chart(fig)  # display plotly chart

            def plot_candle_data():  # function for plotting candle data
                fig = go.Figure()  # create figure
                fig.add_trace(go.Candlestick(x=data['Date'],  # add candlestick plot
                                             # x-axis: date, open
                                             open=data['Open'],
                                             high=data['High'],  # y-axis: high
                                             low=data['Low'],  # y-axis: low
                                             close=data['Close'], name='market data'))  # y-axis: close
                fig.update_layout(  # update layout
                    title='Candlestick Chart of {}'.format(a),  # title
                    yaxis_title='Stock Price',  # y-axis: title
                    xaxis_title='Date')  # x-axis: title
                st.plotly_chart(fig)  # display plotly chart

            chart = ('Candle Stick', 'Line Chart')  # chart types
            # dropdown for selecting chart type
            dropdown1 = st.selectbox('Pick your chart', chart)
            with st.spinner('Loading...'):  # spinner while loading
             time.sleep(2)
            if (dropdown1) == 'Candle Stick':  # if user selects 'Candle Stick'
                plot_candle_data()  # plot candle data
            elif (dropdown1) == 'Line Chart':  # if user selects 'Line Chart'
                plot_raw_data()  # plot raw data
            else:  # if user doesn't select any chart
                plot_candle_data()  # plot candle data

# Real-Time Stock Price Section Ends Here

# Stock Price Prediction Section Starts Here
elif(selected == 'Stock Prediction'):  # if user selects 'Stock Prediction'
    st.subheader("Stock Prediction")

    tickers = stock_df["Company Name"]  # get company names from csv file
    # dropdown for selecting company
    a = st.selectbox('Pick a Company', tickers)
    with st.spinner('Loading...'):  # spinner while loading
             time.sleep(2)
    dict_csv = pd.read_csv('StockStreamTickersData.csv', header=None, index_col=0).to_dict()[1]  # read csv file
    symb_list = []  # list for storing symbols
    val = dict_csv.get(a)  # get symbol from csv file
    symb_list.append(val)  # append symbol to list
    if(a == ""):  # if user doesn't select any company
        st.write("Enter a Stock Name")  # display message
    else:  # if user selects a company
        # download data from yfinance
        data = yf.download(symb_list, start=start, end=end)
        data.reset_index(inplace=True)  # reset index
        st.subheader('Raw Data of {}'.format(a))  # display raw data
        st.write(data)  # display data

        def plot_raw_data():  # function for plotting raw data
            fig = go.Figure()  # create figure
            fig.add_trace(go.Scatter(  # add scatter plot
                x=data['Date'], y=data['Open'], name="stock_open"))  # x-axis: date, y-axis: open
            fig.add_trace(go.Scatter(  # add scatter plot
                x=data['Date'], y=data['Close'], name="stock_close"))  # x-axis: date, y-axis: close
            fig.layout.update(  # update layout
                title_text='Time Series Data of {}'.format(a), xaxis_rangeslider_visible=True)  # title, x-axis: rangeslider
            st.plotly_chart(fig)  # display plotly chart

        plot_raw_data()  # plot raw data
        # slider for selecting number of years
        n_years = st.slider('Years of prediction:', 1, 4)
        period = n_years * 365  # calculate number of days

        # Predict forecast with Prophet
        # create dataframe for training data
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(
            columns={"Date": "ds", "Close": "y"})  # rename columns

        m = Prophet()  # create object for prophet
        m.fit(df_train)  # fit data to prophet
        future = m.make_future_dataframe(
            periods=period)  # create future dataframe
        forecast = m.predict(future)  # predict future dataframe

        # Show and plot forecast
        st.subheader('Forecast Data of {}'.format(a))  # display forecast data
        st.write(forecast)  # display forecast data

        st.subheader(f'Forecast plot for {n_years} years')  # display message
        fig1 = plot_plotly(m, forecast)  # plot forecast
        st.plotly_chart(fig1)  # display plotly chart

        st.subheader("Forecast components of {}".format(a))  # display message
        fig2 = m.plot_components(forecast)  # plot forecast components
        st.write(fig2)  # display plotly chart

# Stock Price Prediction Section Ends Here

#Model
elif(selected == 'LSTM'): 
    st.subheader("LSTM MODEL")
    userinput = st.text_input('Enter Stock Name')
    df = yf.download(userinput, period='10y',interval='1d')
    df.head()

    #Visual
    st.subheader('Closing price vs Time Chart')
    fig0 = plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.plotly_chart(fig0)
    
    st.subheader( 'Closing Price vs Time chart with 100MA' )
    ma100 = df.Close.rolling(100).mean()
    fig1 = plt.figure(figsize = (12, 6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.plotly_chart(fig1)

    st.subheader( 'Closing Price vs Time chart with 100MA & 200MA' )
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig2 = plt.figure(figsize = (12, 6))
    plt.plot(ma100,'r')
    plt.plot(ma200,'g')
    plt.plot(df.Close)
    st.plotly_chart(fig2)

    # splitting data into train and test

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80): int(len(df))])

    print(data_training.shape)
    print(data_testing.shape)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)


    # Load model lstm
    model=load_model('rnn.keras')

    #testing part
    past_100_day = data_training.tail(100)
    final_df = pd.concat ([past_100_day,data_testing], ignore_index = True)
    input_data = scaler.fit_transform(final_df)


    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test , y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)

    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    #final graph
    st.subheader('Prediction vs Original')
    fig3 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Original price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.plotly_chart(fig3)

    
    # Future Prediction

    future_days = 10  # Number of days to predict into the future
    x_input = input_data[-100:]  # Use the last 100 days as input for predicting the next 'future_days'

    future_predictions = []  # List to store future predictions

    for i in range(future_days):
        x_input = x_input.reshape((1, 100, 1))  # Reshape input for the model
        y_pred = model.predict(x_input)  # Predict next day's price
        future_predictions.append(y_pred[0, 0])  # Append the prediction to the list
        x_input = np.append(x_input[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)  # Update input data by shifting one step forward

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    for day, prediction in enumerate(future_predictions, 1):
        print(f"Day {day}: Predicted Price = {prediction}")

    st.write(f"Day {day}: Predicted Price = {prediction}")




    # Model Accuracy
    from sklearn.metrics import mean_squared_error

    # Inverse transform to get the original scale
    y_test_original = y_test / scale_factor
    y_predicted_original = y_predicted / scale_factor

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test_original, y_predicted_original)
    st.write('Mean Squared Error (MSE):', mse)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print('Root Mean Squared Error (RMSE):', rmse)
    st.write('Root Mean Squared Error (RMSE):', rmse)


 
elif(selected == 'Sentiment Analysis'):
  #Replace with your News API key
    NEWS_API_KEY = "20b7b152074e4527b33ee9dccce6db31"

    # Function to fetch news headlines related to a stock
    def fetch_news_headlines(query, language='en', page_size=100):
        url = f"https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP request errors
        data = response.json()
        if 'articles' in data:
            return [article['title'] for article in data['articles']]
        else:
            return []

    # Function to perform sentiment analysis on a given text
    def get_sentiment(text):
        analysis = TextBlob(text)
        # Check if sentiment is positive, negative, or neutral
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    # Function to analyze sentiment of news headlines related to a stock
    def analyze_news_sentiment(stock_name):
        query = f"{stock_name} stock"
        headlines = fetch_news_headlines(query)
        # Initialize sentiment counters
        sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    
        # Analyze sentiment of each headline
        for headline in headlines:
            sentiment = get_sentiment(headline)
            sentiments[sentiment] += 1
    
        # Calculate total number of headlines
        total_headlines = sum(sentiments.values())
    
        # Check if any headlines were found
        if total_headlines > 0:
            # Calculate sentiment percentages
            positive_percent = sentiments['Positive'] / total_headlines * 100
            negative_percent = sentiments['Negative'] / total_headlines * 100
            neutral_percent = sentiments['Neutral'] / total_headlines * 100
           
            # Display sentiment analysis results
            st.write(f"Sentiment Analysis for News Headlines on {stock_name}:")
            st.write(f"Positive: {positive_percent:.1f}%")
            st.write(f"Negative: {negative_percent:.1f}%")
            st.write(f"Neutral: {neutral_percent:.1f}%")
           
        
            # Plot pie chart of sentiment distribution
            labels = list(sentiments.keys())
            sizes = list(sentiments.values())
            plt.figure(figsize=(8, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            plt.title(f'Sentiment Analysis Pie Chart for {stock_name}')
            plt.axis('equal')
            st.pyplot(plt)  # Display the plot using Streamlit
        else:
            st.write(f"No news headlines found for {stock_name}.")

    # Streamlit application
    def main():
        # Title of the application
        st.title("Stock Market Sentiment Analysis")
    
        # Get stock name from user input
        stock_name = st.text_input("Enter the name of the stock:")
    
        # Button to perform sentiment analysis
        if st.button("Analyze Sentiment"):
            if stock_name:
                analyze_news_sentiment(stock_name)
            else:
                st.write("Please enter a stock name.")
    # Run the application
    if __name__ == "__main__":
        main()

