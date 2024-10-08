import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta,date
from concurrent.futures import ThreadPoolExecutor
from yahooquery import Ticker
import numpy as np
from scipy.stats import norm

end=date(2024,10,4)
start=date(2000,10,4)

# print(f"Start Date:{start} -x-x- End Date:{end}")

df_nifty=pd.read_csv('/Users/devanshjoshi/tensorflow-test/Algo Trading/DPA/Nifty 50 Historical Data.csv')
df_gold=pd.read_csv('/Users/devanshjoshi/tensorflow-test/Algo Trading/DPA/XAU_USD.csv')
def parse_dates(date_str):
    date_formats = ['%m/%d/%y', '%m/%d/%Y', '%Y-%m-%d'] ## Check for any of these formats
    for fmt in date_formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    raise ValueError(f"Date format not recognized for: {date_str}")
        
df_nifty['Date']=df_nifty['Date'].apply(parse_dates)
df_nifty=df_nifty[['Date','Open']]
df_gold['Date']=df_gold['Date'].apply(parse_dates)
df_gold=df_gold[['Date','Open']]

def fetch_data(ticker, start_date, end_date, existing_df=None):
    data = Ticker(ticker) ## using yahooquery's Ticker function to fetch data
    try:
        new_data = data.history(start=start_date, end=end_date)
        
        if new_data.empty:
            print("No new data available for the specified date range.")
            return existing_df
        
        new_data = new_data.reset_index()
        new_data['date'] = pd.to_datetime(new_data['date']).dt.tz_localize(None)
        
        new_data= new_data.rename(columns={
            'date': 'Date',
            'open': 'Open',
        })
        
        new_data = new_data[['Date','Open']]
        
        if(existing_df is not None):
            existing_df['Date'] = pd.to_datetime(existing_df['Date']).dt.tz_localize(None)
            combined_df = pd.concat([new_data, existing_df], ignore_index=True)
            combined_df = combined_df.sort_values('Date')
            combined_df = combined_df.reset_index(drop=True)
            return combined_df
        else:
            return new_data
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return existing_df

start_date = df_nifty['Date'][0]+timedelta(days=1)
df_nifty = fetch_data( "^NSEI", start_date, end, df_nifty)
df_ftse=fetch_data('^FTSE', start, end)

def filter_dates(df1, df2, date_column='Date'):
    df1[date_column] = pd.to_datetime(df1[date_column])
    df2[date_column] = pd.to_datetime(df2[date_column])
    
    df1[date_column] = df1[date_column].dt.normalize()
    df2[date_column] = df2[date_column].dt.normalize()
    
    common_dates = set(df1[date_column]).intersection(set(df2[date_column]))
    
    df1_filtered = df1[df1[date_column].isin(common_dates)]
    df2_filtered = df2[df2[date_column].isin(common_dates)]
    
    df1_filtered = df1_filtered.sort_values(date_column)
    df2_filtered = df2_filtered.sort_values(date_column)
    
    df1_filtered = df1_filtered.reset_index(drop=True)
    df2_filtered = df2_filtered.reset_index(drop=True)
    
    return df1_filtered, df2_filtered

df_nifty, df_ftse =filter_dates(df_nifty, df_ftse)
df_gold, df_nifty =filter_dates(df_gold, df_nifty)

def convert_val(df):
    for col in df.columns:
        if col != 'Date':
            df[col] = df[col].replace(',', '', regex=True).astype(float)
    return df

df_nifty=convert_val(df_nifty)
df_gold=convert_val(df_gold)
df_ftse=convert_val(df_ftse)

df_nifty.set_index('Date',inplace=True)
df_ftse.set_index('Date',inplace=True)
df_gold.set_index('Date',inplace=True)
df_merged=pd.concat([df_nifty, df_ftse, df_gold], axis=1, join='inner')
df_merged.columns=['open_nifty', 'open_ftse', 'open_gold']
df_merged['nifty_vol'] = df_merged["open_nifty"].pct_change().rolling(22).std()
df_merged['ftse_vol'] = df_merged["open_ftse"].pct_change().rolling(22).std()
df_merged['gold_vol'] = df_merged["open_gold"].pct_change().rolling(22).std()
df_merged.reset_index(inplace=True)
df_int=pd.read_csv('/Users/devanshjoshi/tensorflow-test/Algo Trading/DPA/interest_rate_data.csv')
df_int['Date']=df_int['Date'].apply(parse_dates)
df_int.tail()
merged_df = pd.merge(df_int, df_merged, on='Date', how='inner')
merged_df['int_nifty']=merged_df['int_nifty']/100
merged_df['int_ftse']=merged_df['int_nifty']/100
merged_df['int_gold']=merged_df['int_nifty']/100


def calculate_black_scholes_options(df):
    def black_scholes_put(S, K, T, r, sigma): ##Function to calculate put option price using BSM Model
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price

    def black_scholes_delta(S, K, T, r, sigma): ##Function to calculate put option delta using BSM Model
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        delta = -norm.cdf(-d1)
        return delta

    T = 5 / 365 ## Options have been assumed to be weekly options

    df['put_price_nifty'] = black_scholes_put(
        S=df['open_nifty'], K=df['open_nifty'], T=T, 
        r=df['int_nifty'], sigma=df['nifty_vol']
    )

    df['put_price_ftse'] = black_scholes_put(
        S=df['open_ftse'], K=df['open_ftse'], T=T, 
        r=df['int_ftse'], sigma=df['ftse_vol']
    )

    df['put_price_gold'] = black_scholes_put(
        S=df['open_gold'], K=df['open_gold'], T=T, 
        r=df['int_gold'], sigma=df['gold_vol']
    )

    df['delta_nifty'] = black_scholes_delta(
        S=df['open_nifty'], K=df['open_nifty'], T=T, 
        r=df['int_nifty'], sigma=df['nifty_vol']
    )

    df['delta_ftse'] = black_scholes_delta(
        S=df['open_ftse'], K=df['open_ftse'], T=T, 
        r=df['int_ftse'], sigma=df['ftse_vol']
    )

    df['delta_gold'] = black_scholes_delta(
        S=df['open_gold'], K=df['open_gold'], T=T, 
        r=df['int_gold'], sigma=df['gold_vol']
    )

    return df

merged_df=calculate_black_scholes_options(merged_df)
merged_df=merged_df[['Date','open_nifty', 'open_ftse',
       'open_gold','put_price_nifty',
       'put_price_ftse', 'put_price_gold', 'delta_nifty', 'delta_ftse',
       'delta_gold']]
merged_df
