import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import time

def scrape_aaii():
    # open the old data file
    old_df = pd.read_csv('repoData/aaii_sentiment.csv')
    # get the last date in the old data
    last_date = pd.to_datetime(old_df['Date'].iloc[-1]).date()
    if (last_date + pd.DateOffset(days=5)).date() >= pd.Timestamp.today().date():
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

        url = "https://www.aaii.com/sentimentsurvey/sent_results"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # find the sentiment table
            table = soup.find('table')
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
    # remove the % sign and convert to float
    # add the bull bear difference
    df['Bull-Bear Spread'] = (df['Bullish'].str.replace('%', '').astype(float) - df['Bearish'].str.replace('%', '').astype(float)) / 100
    # drop the bullish, neutral and bearish columns
    df.drop(columns=['Bullish', 'Neutral', 'Bearish'], inplace=True)
    # rename the date column
    df.rename(columns={'Reported Date': 'Date'}, inplace=True)
    # convert to datetime
    # date is in the format of "MonthName Day" so we need to add the year which is the current year except the current date without year is lower
    # than the current date, in that case we add the previous year
    today = pd.Timestamp.today()
    df['Date'] = df['Date'].apply(lambda x: pd.to_datetime(x + ' ' + str(today.year)).date() if pd.to_datetime(x + ' ' + str(today.year)) < today else pd.to_datetime(x + ' ' + str(today.year - 1)).date())
    df['Date'] = df['Date'] + pd.DateOffset(days=1)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    # reverse the table
    df = df.iloc[::-1]
    # get the new data
    new_df = df[df['Date'] > last_date]
    # append the new data to the old data
    new_df = pd.concat([old_df, new_df], ignore_index=True)
    # save the new data
    new_df.to_csv('repoData/aaii_sentiment.csv', index=False)


if __name__ == "__main__":
    scrape_aaii()