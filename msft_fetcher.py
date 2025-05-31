import yfinance as yf

ticker_symbol = 'MSFT'
msft = yf.Ticker(ticker_symbol)

start_date = '1986-03-13'
end_date = '2025-04-28'

try:
    historical_data = msft.history(start=start_date, end=end_date, interval='1d')
    
    if not historical_data.empty:
        historical_data.to_csv('MSFT_Historical_Data.csv')
        print(f"Data downloaded successfully! {len(historical_data)} records saved as 'MSFT_Historical_Data.csv'")
    else:
        print("No data retrieved. Please check the ticker symbol and date range.")
        
except Exception as e:
    print(f"An error occurred: {e}")