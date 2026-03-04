# script to download security prices

import os
import traceback

import yfinance as yf

if __name__ == '__main__':
    """after downloading the data, run script data/clean_yfinance_data.sh."""
    try:
        data_dir = os.path.join(os.getcwd(), "data")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        security_list = []

        # current focus
        security_list += ['GLD', 'NVS', 'PLTR']

        # basic ETFs
        #security_list = ['SPY', 'VTV', 'VBR', 'VTI', 'VWO', 'BND', 'AGG', 'JNK', 'TIP', 'VNQ', 'GSG', 'DBC']

        # additional ETF
        #security_list += ['QQQ', 'LQD', 'HYG', 'SOXX', 'SOXL', 'TLT', 'GLD', 'IBIT']

        period = '20y'
        interval = '1d'

        for ticker in security_list:
            df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
            #df = yf.download(ticker, start="2005-01-01", end="2025-04-22", auto_adjust=False)
            df['SEC_ID'] = ticker
            print(df.head())
            df.to_csv(os.path.join(data_dir, ticker + '.csv'))

    except Exception as err:
        print('Data downloading batch failed: ' + str(err))
        print(traceback.format_exc())