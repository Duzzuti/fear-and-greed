import yfinance as yf
import yfinance.exceptions as yf_exc
import yfinance.shared as yf_shared

INVALID_TICKER = "INVALID TICKER"
NO_DATA_IN_RANGE = "NO DATA IN RANGE"

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
            error_type = error.split("('")[0]
            error_msg = error.split("('")[1][:-2]
        except Exception as e:
            print("Could not parse error message: " + error)
            exit()
        try:
            full_err = getattr(yf_exc, error_type)(error_msg)
        except AttributeError as e:
            print("Unknown exception: " + error)
            exit()
        if full_err.__class__ in (yf_exc.YFTzMissingError, yf_exc.YFTickerMissingError):
            err = INVALID_TICKER
        elif full_err.__class__ == yf_exc.YFPricesMissingError:
            err = NO_DATA_IN_RANGE
        else:
            print(f"Unhandled error occurred: {full_err}")
            exit()
    return (data, err)