# test_yfinance.py - Test script to debug yfinance data structure
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Test downloading data
print("Testing yfinance data download...")
print("-" * 50)

# Test parameters
stock1 = "MSFT"
stock2 = "GOOGL"
start_date = datetime.now() - timedelta(days=100)
end_date = datetime.now()

print(f"Downloading data for {stock1} and {stock2}")
print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print("-" * 50)

# Method 1: Download both together
print("\nMethod 1: Download both stocks together")
try:
    data = yf.download([stock1, stock2], start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Column levels: {data.columns.nlevels}")
    if data.columns.nlevels > 1:
        print(f"Level 0 values: {data.columns.get_level_values(0).unique().tolist()}")
        print(f"Level 1 values: {data.columns.get_level_values(1).unique().tolist()}")
    print("\nFirst few rows:")
    print(data.head())
except Exception as e:
    print(f"Error: {e}")

# Method 2: Download separately
print("\n" + "-" * 50)
print("\nMethod 2: Download stocks separately")
try:
    data1 = yf.download(stock1, start=start_date, end=end_date, auto_adjust=True)
    data2 = yf.download(stock2, start=start_date, end=end_date, auto_adjust=True)
    print(f"\n{stock1} data shape: {data1.shape}")
    print(f"{stock1} columns: {data1.columns.tolist()}")
    print(f"\n{stock2} data shape: {data2.shape}")
    print(f"{stock2} columns: {data2.columns.tolist()}")
    
    # Combine data
    combined = pd.DataFrame({
        stock1: data1['Close'],
        stock2: data2['Close']
    })
    print(f"\nCombined data shape: {combined.shape}")
    print(f"Combined columns: {combined.columns.tolist()}")
    print("\nFirst few rows of combined data:")
    print(combined.head())
except Exception as e:
    print(f"Error: {e}")

print("\n" + "-" * 50)
print("Test complete!")
print("\nRecommendation: Use the working method in your app.py")