import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time

def scrape_put_call():
    # open the old data file
    old_df = pd.read_csv('repoData/put_call_ratios.csv')
    # get the last date in the old data
    last_date = pd.to_datetime(old_df['Date'].iloc[-1]).date()
    if last_date >= pd.Timestamp.today().date():
        print("No new data available.")
        return
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
                f'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q={q}',
                'application/json,text/plain,*/*',
                'text/csv,text/plain'
            ]),
            'Accept-Language': random.choice([f'en-US,en;q={q}', f'fr-FR,fr;q={q}', f'de-DE,de;q={q}']),
            'Accept-Encoding': random.choice(['gzip, deflate', 'gzip, deflate, br']),
            'Connection': 'keep-alive',
        }

        url = "https://ycharts.com/indicators/cboe_equity_put_call_ratio"
        response = requests.get(url, headers=headers)
        #print(response.text)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # find the sentiment tables
            table1 = soup.find_all('table')[5]
            table2 = soup.find_all('table')[6]
            if table1 is not None and table2 is not None:
                break
        
        print(f"({i})Failed to access the page or find the table. Status code:", response.status_code)
        time.sleep(random.randint(2, 5))
        if i == max_tries - 1:
            raise Exception("Failed to access the page after", max_tries, "tries. Exiting...")
        continue
    
    # process the table
    rows = []
    for row in table1.find_all('tr'):
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        rows.append(cells)
    for row in table2.find_all('tr')[1:]:
        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
        rows.append(cells)

    df = pd.DataFrame(rows[1:], columns=rows[0])
    # convert the date to datetime, format is "fullMonthName day, year"
    df['Date'] = pd.to_datetime(df['Date'], format='%B %d, %Y') + pd.DateOffset(days=1)
    df['Date'] = df['Date'].dt.date
    #rename Value to PCR
    df.rename(columns={'Value': 'PCR'}, inplace=True)
    # reverse the table
    df = df.iloc[::-1]
    # get the new data
    new_df = df[df['Date'] > last_date]
    # append the new data to the old data
    new_df = pd.concat([old_df, new_df], ignore_index=True)
    # save the new data
    new_df.to_csv('repoData/put_call_ratios.csv', index=False)


if __name__ == "__main__":
    scrape_put_call()