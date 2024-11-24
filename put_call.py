import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_put_call_ratios(dir):
    # try to get the old data
    try:
        data = pd.read_csv(dir + "put_call_ratios.csv", index_col=0, parse_dates=True)
        return data
    except FileNotFoundError:
        cookies = {'session': os.getenv("ALPHALERTS_SESSION_COOKIE")}
        r = requests.get(
            'https://alphalerts.com/live-historical-equity-pcr/download',
            cookies=cookies
        )
        # save the response to a file
        with open(dir + "put_call_ratios.csv", "w") as f:
            f.write(r.content.decode().strip("\n"))

        # remove every second line
        with open(dir + "put_call_ratios.csv", "r") as f:
            lines = f.readlines()
        with open(dir + "put_call_ratios.csv", "w") as f:
            for i, line in enumerate(lines):
                if i % 2 == 0:
                    f.write(line)

        # convert the timestamps to Date Objects
        data = pd.read_csv(dir + "put_call_ratios.csv", header=0)
        # get first column converted to datetime objects from unix timestamps
        data["Date"] = pd.to_datetime(data.iloc[:, 0] / 1000, unit="s")
        data.set_index("Date", inplace=True)
        #remove the first column
        data = data.iloc[:, 1:]
        data.to_csv(dir + "put_call_ratios.csv")
        return data