import pandas as pd

def scrape(data_dir):
    # Scrape the Wikipedia page for S&P 500 companies
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)[0]  # Read the first table on the page

    # Save to a CSV if needed
    sp500_table["Symbol"].to_csv(data_dir + "sp500_companies.csv", index=False)