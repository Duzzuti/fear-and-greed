from sys import exit
import pandas as pd
import yfinance as yf
import yfinance.exceptions as yf_exc
import yfinance.shared as yf_shared
from time import sleep
from requests.sessions import Session

INVALID_TICKER = "INVALID TICKER"
NO_DATA_IN_RANGE = "NO DATA IN RANGE"
TIMEOUT = "TIMEOUT"
DEBUG_ERROR = "DEBUG ERROR"
MAX_ERR_COUNT = 5

def isDataInRange(start, end, trading_days):
    if start == None:
        return True
    start_date = pd.to_datetime(start).strftime("%Y-%m-%d")
    if isinstance(trading_days, pd.DataFrame) and not trading_days.empty:
        # check if there are trading days in the range
        if end == None:
            data_range = trading_days.loc[start_date:]
        else:
            data_range = trading_days.loc[start_date:(pd.to_datetime(end) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")]
        if data_range.empty:
            return False
    else:
        err1 = DEBUG_ERROR
        while err1 != TIMEOUT:
            _, err1 = downloadWithExceptions("SPY", start=start, end=end)
            if err1 == TIMEOUT:
                print(f"Error: Timeout for SPY. Retrying...")
        if err1 == NO_DATA_IN_RANGE:
            return False
    return True

def downloadWithExceptions(ticker : str, start=None, end=None):
    ticker = ticker.upper()
    data = yf.download(ticker, start=start, end=end)
    err = None
    if yf_shared._ERRORS:
        # some exceptions occurred, get the error
        if not yf_shared._ERRORS[ticker]:
            # should not happen
            raise ValueError("Got an exception for a different ticker: " + str(yf_shared._ERRORS) + " for ticker " + ticker)
        error = yf_shared._ERRORS[ticker]
        try:
            if error.startswith("ReadTimeout") or error.startswith("ConnectionError") or error.startswith("ChunkedEncodingError"):
                error_type = "ReadTimeout"
                error_msg = error.split('("')[1][:-2]
            else:
                error_type = error.split("('")[0]
                error_msg = error.split("('")[1][:-2]
        except Exception as e:
            print("Could not parse error message: " + error)
            exit()
        try:
            if error_type in (yf_exc.YFDataException.__name__, yf_exc.YFException.__name__, yf_exc.YFNotImplementedError.__name__):
                full_err = getattr(yf_exc, error_type)(error_msg)
            elif error_type in (yf_exc.YFEarningsDateMissing.__name__, yf_exc.YFTzMissingError.__name__):
                full_err = getattr(yf_exc, error_type)(ticker)
            elif error_type in (yf_exc.YFPricesMissingError.__name__, yf_exc.YFTickerMissingError.__name__):
                full_err = getattr(yf_exc, error_type)(ticker, error_msg)
            elif error_type == yf_exc.YFInvalidPeriodError.__name__:
                full_err = getattr(yf_exc, error_type)(ticker, "start=" + str(start) + ", end=" + str(end), error_msg)
            elif error_type in ["ReadTimeout", "JSONDecodeError"]:
                full_err = Exception(error_msg)
            else:
                raise AttributeError("Unknown error type: " + error_type)
        except AttributeError as e:
            print("Unknown exception: " + error)
            exit()
        if full_err.__class__ in (yf_exc.YFTzMissingError, yf_exc.YFTickerMissingError):
            err = INVALID_TICKER
        elif full_err.__class__ == yf_exc.YFPricesMissingError:
            err = NO_DATA_IN_RANGE
        elif full_err.__class__ == Exception:
            err = TIMEOUT
        else:
            print(f"Unhandled error occurred: {full_err}")
            exit()
    return (data, err)

def downloadCompleteHandler(ticker : str, start=None, end=None, ignore_no_data=False, check_replacement_first=False, trading_days=None):
    if isinstance(trading_days, pd.DataFrame) and not trading_days.empty:
        if not isDataInRange(start, end, trading_days):
            return pd.DataFrame()
    start_ticker = ticker
    if check_replacement_first:
        with open("repoData/replacement_list.csv", "r") as f:
            replacements = f.readlines()
        for replacement in replacements:
            old, new_ticker = replacement.strip().split(",")
            if old == ticker:
                if new_ticker:
                    ticker = new_ticker
                else:
                    return None
                break
    err = DEBUG_ERROR
    err_count = 0
    while err:
        err_count += 1
        if err_count >= 2:
            print(f"Error: Failed downloading data for {ticker} twice. Resetting yfinance.")
            yf.utils._requests = Session()
        if err_count > MAX_ERR_COUNT:
            print(f"Error: Failed downloading data for {ticker} {MAX_ERR_COUNT} times. Exiting.")
            exit()
        data, err = downloadWithExceptions(ticker, start=start, end=end)
        if err == TIMEOUT:
            print(f"Error: Timeout for {ticker}. Retrying...")
            continue
        if err and "." in ticker:
            # try replacing the dot with dash
            data, err = downloadWithExceptions(ticker.replace(".", "-"), start=start, end=end)
            if err == TIMEOUT:
                print(f"Error: Timeout for {ticker}. Retrying...")
                continue
        if err == INVALID_TICKER or (ignore_no_data and err == NO_DATA_IN_RANGE):
            if start_ticker != ticker:
                print(f"Error: {ticker} is not a valid ticker replacement for {start_ticker}.")
                # remove from replacement list
                with open("repoData/replacement_list.csv", "r") as f:
                    replacements = f.readlines()
                with open("repoData/replacement_list.csv", "w") as f:
                    for replacement in replacements:
                        old, new_ticker = replacement.strip().split(",")
                        if old != start_ticker:
                            f.write(f"{old},{new_ticker}\n")

            # check replacement list for new ticker
            with open("repoData/replacement_list.csv", "r") as f:
                replacements = f.readlines()
            for replacement in replacements:
                old, new_ticker = replacement.strip().split(",")
                if old == ticker:
                    # if replacement does not exist, we should skip this data
                    if not new_ticker:
                        return None
                    break
            else:
                if err_count < MAX_ERR_COUNT:
                    if ignore_no_data and err == NO_DATA_IN_RANGE:
                        if not isDataInRange(start, end, trading_days):
                            break
                    print(f"Error: Failed downloading data for expected valid ticker: {ticker}. Retrying...{err_count}")
                    sleep(10)
                    continue
                new_ticker = None
                if ignore_no_data and err == NO_DATA_IN_RANGE:
                    print(f"Error: No data for {ticker} in range {start} to {end}. Add to replacement list.")
                else:
                    print(f"Error: Invalid ticker: {ticker}. Add to replacement list.")
                while not new_ticker or new_ticker == ticker:
                    if ignore_no_data:
                        with open("test_data/sp500_possible_replacements.csv", "r") as f:
                            content = f.readlines()
                        for line in content:
                            if line.startswith(ticker):
                                print(f"Possible replacements for {ticker}: {','.join(line.split(',')[1:])}")
                                break
                    new_ticker = input("Enter new (correct) ticker, 'SKIP' to ignore ths ticker or 'QUIT' to exit: ")
                    if new_ticker == "QUIT":
                        exit()
                    elif new_ticker == "SKIP":
                        return data
                with open("repoData/replacement_list.csv", "a") as f:
                    f.write(f"{ticker},{new_ticker}\n")
                print("Replacement list updated.")
            ticker = new_ticker
        elif err == NO_DATA_IN_RANGE:
            if isDataInRange(start, end, trading_days):
                print(f"Error: No data for {ticker} in range {start} to {end}.")
                exit()
            else:
                break
        err_count = 0
    return data

    