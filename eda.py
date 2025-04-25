# Exploratory Data Analysis

# python standard library: https://docs.python.org/3/library/
import os
import traceback

# third party package
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

seaborn.set()

if __name__ == '__main__':
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        data_dir = os.path.join(os.getcwd(), 'data', 'archived')

        # load data

        df = pd.read_csv(os.path.join(data_dir, 'frc.csv'))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        print(f"""--- data start timestamp: {min(df.index)}, data end start timestamp: {max(df.index)}""")
        print(df.describe())

        # price and volume evolution

        top_plt = plt.subplot2grid((5, 4), (0, 0), rowspan=3, colspan=4)
        top_plt.plot(df.index, df["Adj Close"])
        plt.title('Historical stock prices of First Republic Bank')

        bottom_plt = plt.subplot2grid((5, 4), (3, 0), rowspan=1, colspan=4)
        bottom_plt.plot(df.index, np.log(df['Volume']), color='red')
        plt.title('\nFirst Republic Bank Trading Volume (Log Scale)', y=-0.60)

        plt.gcf().set_size_inches(12, 8)

        # Using moving average to identify regimes

        # regime changes in stock price evolution
        rolling = df['Adj Close'].rolling(window=250, center=True)
        data = pd.DataFrame(
            {'input': df['Adj Close'], 'one-year rolling mean': rolling.mean(), 'one-year rolling std': rolling.std()})
        ax = data.plot(style=['-', '--', ':'])

        plt.vlines(x=[pd.to_datetime('2/8/2017'), pd.to_datetime('1/14/2020'), pd.to_datetime('11/16/2021')], ymin=0,
                   ymax=df['Adj Close'].max(), linestyles='dashed', colors='green')

        ax.lines[0].set_alpha(0.3)

        plt.gcf().set_size_inches(12, 8)

        df = pd.read_csv(os.path.join(data_dir, 'gme.csv'))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[df.index >= pd.to_datetime('7/1/2019')]

        # regime changes in stock price evolution
        rolling = df['Adj Close'].rolling(window=250, center=True)
        data = pd.DataFrame(
            {'input': df['Adj Close'], 'one-year rolling mean': rolling.mean(), 'one-year rolling std': rolling.std()})
        ax = data.plot(style=['-', '--', ':'])

        ax.lines[0].set_alpha(0.3)

        plt.gcf().set_size_inches(12, 8)
    except Exception as err:
        print('LPPL run failed: ' + str(err))
        print(traceback.format_exc())
