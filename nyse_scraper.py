import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL for the screener (NYSE example)
base_url = "https://finance.yahoo.com/screener/213b0631-0f1f-4ff6-bf22-6d06adc8ed54?count=100&offset="

# Data storage
all_data = []

# Iterate over paginated results
offset = 0
while True:
    print(f"Scraping offset {offset}...")
    url = base_url + str(offset)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract table rows
    table = soup.find('table')
    if not table:
        print("No more data found.")
        break

    rows = table.find_all('tr')[1:]  # Skip header row
    if not rows:
        print("No more rows found.")
        break

    for row in rows:
        cols = row.find_all('td')
        data = [col.text.strip() for col in cols]
        all_data.append(data)
    
    # Move to the next page
    offset += 100

# Convert to DataFrame
columns = ["Symbol", "Name", "Last Price", "Market Time", "Change", "Change %", "Volume", "Avg Volume", "Market Cap", "PE Ratio"]
df = pd.DataFrame(all_data, columns=columns)

# Save to CSV
df.to_csv("nyse_screener.csv", index=False)
print("Data saved to nyse_screener.csv")