import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
import calendar

def scrape_insider():
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

        url = "https://www.gurufocus.com/economic_indicators/4359/insider-buysell-ratio-usa-overall-market"
        response = requests.get(url, headers=headers)
        time.sleep(1)
        response = requests.get(url, headers=headers)
        print(headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text.encode('utf-8'), 'html.parser')
            # find the insider transaction table
            if len(soup.find_all('table')) < 2:
                print(f"({i})Failed to find the table. Retrying...")
                time.sleep(random.randint(2, 5))
                continue
            table = soup.find_all('table')[1]
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
    # get the date of the data
    date = df['Value'].iloc[1]
    date = pd.to_datetime(date, format='%Y-%m-%d', errors='coerce').date()
    # get value
    value = df['Value'].iloc[0]
   
    # open the old data file and appending possible new data
    old_df = pd.read_csv('repoData/insider.csv')
    # get the last date in the old data
    last_date = pd.to_datetime(old_df['Date'].iloc[-1]).date()
    # if months are equal add the data if it is different
    if not(last_date.month == date.month and float(value) == float(old_df['Value'].iloc[-1])):
        new_df = pd.concat([old_df, pd.DataFrame({"Date": pd.Timestamp.today().date(), "Value": value}, index=["Date"])], ignore_index=True)
        # save the new data
        new_df.to_csv('repoData/insider.csv', index=False)
    else:
        print("The data is already up-to-date.")

if __name__ == "__main__":
    scrape_insider()