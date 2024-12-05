import pandas as pd
import datetime as dt

def scrape_companies(data_dir="repoData/"):
    # Scrape the Wikipedia page for S&P 500 companies
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)[0]  # Read the first table on the page

    current_df = pd.read_csv(data_dir + "sp500_companies.csv")
    # look at last entry in the current_df
    last_entry = current_df.iloc[-1]
    last_entry_date = dt.datetime.strptime(last_entry["date"], "%Y-%m-%d").date()
    current_date = dt.datetime.now().date()
    last_entry_companies = sorted(last_entry["tickers"].split(","))
    current_companies = sorted(sp500_table["Symbol"].tolist())

    additions = set(current_companies) - set(last_entry_companies)
    removals = set(last_entry_companies) - set(current_companies)

    if last_entry_companies == current_companies:
        print("No new changes in S&P 500 companies")
        return
    elif last_entry_date == current_date:
        # need to update the last entry
        current_df.iloc[-1] = [current_date, ",".join(current_companies)]
        print("Updated the last entry. Added: ", additions, " Removed: ", removals)
    else:
        # add a new entry
        new_entry = pd.DataFrame({"date": current_date, "tickers": ",".join(current_companies)}, index=[0])
        current_df = pd.concat([current_df, new_entry], ignore_index=True)
        print("Added a new entry. Added: ", additions, " Removed: ", removals)
    # Save the table to a CSV file
    current_df.to_csv(data_dir + "sp500_companies.csv", index=False)

if __name__ == "__main__":
    scrape_companies()