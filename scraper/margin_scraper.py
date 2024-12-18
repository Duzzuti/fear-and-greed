import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random
import time

def scrape_margin_stats():
    # data fetching
    max_tries = 5
    for i in range(max_tries):
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36',
            'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'
        ]
        q = round(random.random(), 1)
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': random.choice([
                f'text/html,text/javascript,application/xhtml+xml,application/xml;q={q},*/*;q={q}',
                'text/html,text/javascript,application/json,text/plain,*/*',
                'text/html,text/javascript,text/csv,text/plain'
            ]),
            'Accept-Language': random.choice([f'en-US,en;q={q}', f'fr-FR,fr;q={q}', f'de-DE,de;q={q}']),
            'Accept-Encoding': random.choice(['gzip, deflate', 'gzip, deflate, br']),
            'Connection': 'keep-alive',
        }

        url = "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics"
        response = requests.get(url, headers=headers)
        # time.sleep(1)
        # response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text.encode('utf-8'), 'html.parser')
            # find the margin stats table
            table = soup.find_all('table')[0]
            if table is not None:
                break
        
        print(f"({i})Failed to access the page or find the table. Status code:", response.status_code)
        time.sleep(random.randint(2, 5))
        if i == max_tries - 1:
            raise Exception("Failed to access the page after", max_tries, "tries. Exiting...")
        continue
    
    # process the table
    rows = []
    for row in table.find_all('tr'):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        rows.append(cells)
    
    df = pd.DataFrame(rows[1:], columns=rows[0])
    # set first column as index
    df.set_index(df.columns[0], inplace=True)
    df.index = df.index.rename('Date')
    # format example: Aug-24
    df.index = pd.to_datetime(df.index, format='%b-%y', errors='coerce').date
    df = df[::-1]
    #rename columns
    df.rename(columns={"Debit Balances in Customers' Securities Margin Accounts": 'Debit'}, inplace=True)
    df.rename(columns={"Free Credit Balances in Customers' Securities Margin Accounts": 'Credit S'}, inplace=True)
    df.rename(columns={"Free Credit Balances in Customers' Cash Accounts": 'Credit C'}, inplace=True)
    # convert columns to numeric (remove commas)
    df = df.apply(lambda x: x.str.replace(',', ''))
    df = df.apply(pd.to_numeric)
    df["Credit"] = df["Credit S"] + df["Credit C"]
    df.drop(columns=["Credit S", "Credit C"], inplace=True)
    df["Leverage Ratio"] = df["Debit"] / df["Credit"]
    df["Leverage Ratio"] -= 1.5
    df["Leverage Ratio"] = (np.tanh(df["Leverage Ratio"]*2) + 1) *50
    # open the old data file and appending possible new data
    old_df = pd.read_csv('repoData/margin_stats.csv', index_col=0, parse_dates=True)
    old_df.index = old_df.index.date
    old_df.index.rename('Date', inplace=True)
    # get the last date in the old data
    last_date = old_df.index[-1]
    # get the new data
    new_df = df[df.index > last_date]
    new_df.index.rename('Date', inplace=True)
    # append the new data to the old data
    new_df = pd.concat([old_df, new_df])
    # save the new data
    new_df.to_csv('repoData/margin_stats.csv')

if __name__ == "__main__":
    scrape_margin_stats()
