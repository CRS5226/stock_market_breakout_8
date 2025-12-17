import yfinance as yf

symbol = "MOTHERSON.NS"


def get_realtime_data():
    ticker = yf.Ticker(symbol)

    # Correct intraday request format
    data = ticker.history(
        period="1d", interval="1m"  # 1 day of data  # 1-minute candles
    )

    if data.empty:
        print("No data received. Check ticker or market timing.")
        return

    last_row = data.iloc[-1]

    print("\n===========================")
    print(f"Symbol: {symbol}")
    print(f"Timestamp: {last_row.name}")
    print(f"Open:   {last_row['Open']}")
    print(f"High:   {last_row['High']}")
    print(f"Low:    {last_row['Low']}")
    print(f"Close:  {last_row['Close']}")
    print(f"Volume: {last_row['Volume']}")
    print("===========================\n")


# Run once for testing
get_realtime_data()
