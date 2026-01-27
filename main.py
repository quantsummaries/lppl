import os
import traceback

import pandas as pd

from lppl import LPPLModel


def main():
    print("Hello from lppl!")
    try:
        pd.set_option('display.width', 400)
        pd.set_option('display.max_columns', 20)

        data_dir = os.path.join(os.getcwd(), 'data')

        model = LPPLModel(data_path=os.path.join(data_dir, 'GLD.csv'))

        model.predict(start_dt=pd.to_datetime('11/1/2025'),
                      end_dt=pd.to_datetime('1/25/2026'),
                      peak_dt=None)

    except Exception as err:
        print('LPPL run failed: ' + str(err))
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
