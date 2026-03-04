import os
import traceback
from datetime import date

import pandas as pd

from lppl import LPPLModel


def main():
    print("Hello from lppl!")
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        data_dir = os.path.join(os.getcwd(), 'data')

        tickers = ['GLD', 'NVS', 'PLTR']
        calib_times = ['11/1/2025', '11/16/2025', '2/8/2026']
        for ticker, calib_time in zip(tickers, calib_times):
            model = LPPLModel(data_path=os.path.join(data_dir, f'{ticker}.csv'))

            model.predict(start_dt=pd.to_datetime(calib_time),
                          end_dt=pd.to_datetime(date.today()),
                          peak_dt=None)

    except Exception as err:
        print('LPPL run failed: ' + str(err))
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
