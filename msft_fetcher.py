import yfinance as yf

# Define the ticker symbol
ticker_symbol = 'MSFT'


msft = yf.Ticker(ticker_symbol)

# Set date range 
start_date = '1986-03-13'
end_date = '2025-04-28'


historical_data = msft.history(start=start_date, end=end_date, interval='1d')

historical_data.to_csv('MSFT_Historical_Data.csv')

print("Data downloaded and saved as 'MSFT_Historical_Data.csv'")
