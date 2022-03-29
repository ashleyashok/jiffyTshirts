import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

def load_df(filename):
    """
    Loads the dataframe from the CSV file.
    """
    df = pd.read_csv(filename)
    return df

def join_df(df1,df2, on):
    """
    Joins two dataframes on the specified column.
    """
    df = pd.merge(df1,df2,on=on)
    return df

def plot_timeseries(x,y):
    """
    Plot date on x axis and revenue value on y axis.
    """
    plt.plot(x, y)
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.show()
    
    return

def lstm_split_sequence(sequence, n_steps_in, n_steps_out):
    """
    Splits the sequence into subsequences of length n_steps in and n_steps_out.
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # Find the end of this pattern.
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Check if we are beyond the sequence.
        if out_end_ix > len(sequence):
            break
        # Gather data.
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
    

def lstm_model(X,y,train_size,time_steps, batch_size):
    """
    Creates the input data for the LSTM model.
    """
    # Reshape the data to fit the LSTM model.
    data_x = X.reshape((X.shape[0], X.shape[1], 1))
    data_y = y
    # Split the data into training and testing sets.
    x_train = data_x[:int(len(data_x) * train_size)]
    y_train = data_y[:int(len(data_y) * train_size)]
    x_test = data_x[int(len(data_x) * train_size):]
    y_test = data_y[int(len(data_y) * train_size):]
    # Create the LSTM model.
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(time_steps, 1)),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(30, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
    ])
    # Compile the model.
    model.compile(optimizer='adam', loss='mse')
    # Train the model.
    model.fit(x_train, y_train, epochs=3000, batch_size=batch_size, validation_data=(x_test, y_test))
    return model

def get_prediction(model, x):
    """
    Forecasts the LSTM model.
    """
    x_predict = x[-1]
    y_pred = model.predict(x_predict.reshape(1, x_predict.shape[0], 1))
    return y_pred

def plot_two_timeseries(x,y,x_pred,y_pred, title):
    """
    Plots the two time series.
    """
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(x, y, label='Actual')
    ax.plot(x_pred, y_pred, label='Predicted')
    ax.legend()
    fig.suptitle(title)
    fig.savefig("{}.png".format(title))
    fig.show()
    return


if __name__ == "__main__":

    # Load the data.
    customers_df = load_df('data\customers.csv')
    orders_df = load_df('data\orders.csv')
    line_df = load_df('data\line_items.csv')
    # join dataframes
    df1 = join_df(orders_df, line_df, 'order_id')
    df2 = join_df(df1, customers_df, 'customer_uid')
    # create revenue, cost and profit columns
    df2['selling_rev'] = df2['selling_price'] * df2['quantity'] * (1-df2['discount'])
    df2['selling_cost'] = df2['supplier_cost'] * df2['quantity']
    df2['profit'] = df2['selling_rev'] - df2['selling_cost']
    # Change column to datetime type
    df2['order_timestamp'] = pd.to_datetime(df2['order_timestamp'])
    # Change datetime column to date in format YYYY-MM-DD.
    df2['date'] = df2['order_timestamp'].dt.strftime('%Y-%m-%d')
    # Create a new dataframe with the group by date and revenue columns and sort by date.
    df3 = df2.groupby(['date'])['selling_rev'].sum().reset_index()
    df3 = df3.sort_values(by=['date'])


    print(df3.head())
    # Plot the timeseries.
    plot_timeseries(df3['date'], df3['selling_rev'])

    # split sequence into samples
    X,y = lstm_split_sequence(df3['selling_rev'].values, 30, 30)
    model = lstm_model(X,y,train_size=0.8, time_steps=30, batch_size=32)
    
    results = get_prediction(model, x=X)
    
    # Create new dataframe with last 30 rows in df3 and change date column to datetime type
    df4 = df3.tail(30)
    df4['date'] = pd.to_datetime(df4['date'])
    last_date = df4['date'].max()
    
    # create list with 30 days from last date
    date_list = [last_date + datetime.timedelta(days=x) for x in range(1,31)]
    # create dataframe with date_list and results with date as index
    df5 = pd.DataFrame(date_list, columns=['date'])
    df5['date'] = pd.to_datetime(df5['date'])
    df5['prediction'] = results[0].reshape(1,-1)[0]

    # Plot the timeseries.
    plot_two_timeseries(df4['date'], df4['selling_rev'], df5['date'], df5['prediction'], title= 'Daily Revenue Forecast POC Results')
    
    # save model
    model.save('lstm_model.h5')

        
    
