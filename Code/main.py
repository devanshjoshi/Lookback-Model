from Data import merged_df as df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta,date
import os
import base64
import io
import math

def calculations(df):
    df.set_index('Date',inplace=True)## Required to use the .resample function
    df['Returns_nifty']=df['open_nifty'].pct_change().dropna() ##Find daily returns of nifty
    df['Returns_ftse']=df['open_ftse'].pct_change().dropna() ##Find daily returns of ftse
    df['Returns_gold']=df['open_gold'].pct_change().dropna() ##Find daily returns of gold
    qtr_ret_nifty =(1+df['Returns_nifty']).resample('Q').prod()-1 ##Find quarterly returns of nifty for each of the qtrs
    qtr_ret_ftse =(1+df['Returns_ftse']).resample('Q').prod()-1 ##Find quarterly returns of ftse for each of the qtrs
    qtr_ret_gold =(1+df['Returns_gold']).resample('Q').prod()-1 ##Find quarterly returns of gold for each of the qtrs
    df.reset_index(inplace=True)
    df.dropna(inplace=True)
    return qtr_ret_nifty[:-1],qtr_ret_ftse[:-1],qtr_ret_gold[:-1]

ret_nifty,ret_ftse,ret_gold=calculations(df)

def rank_returns(nifty, ftse, gold): ## findthe two assets that performed best and second best for each qtr and assign 0.67 and
    ## 0.33 weights respectively
    df = pd.DataFrame({
        'nifty': nifty,
        'ftse': ftse,
        'gold': gold
    })
    def assign_values(row):
        sorted_indices = row.argsort()
        result = pd.Series(0.0, index=row.index)
        result.iloc[sorted_indices[-1]] = 0.67
        result.iloc[sorted_indices[-2]] = 0.33 
        return result
    result_df = df.apply(assign_values, axis=1)
    return result_df
result = rank_returns(ret_nifty, ret_ftse, ret_gold) ##finding which asset was the best and second best for each qtr
result.reset_index(inplace=True)
def find_columns(row):
    col_067 = row.eq(0.67).idxmax()
    col_033 = row.eq(0.33).idxmax()
    return pd.Series({'col_067': col_067, 'col_033': col_033})
res = result.apply(find_columns, axis=1) ##finding name of 0.67 alloc column and 0.33 alloc column

pd.options.mode.chained_assignment = None ##To avoid settingwithoutcopy warning

last_eq_return=[] ##Stores the value of initial capital at the end of each quarter
opt_chain={} ## stores put options and their respective positions in order to calculate PnL at the end of the each week.

def settle_opt(u67,u33):
    PnL=0.0
    global opt_chain
    for key,val in opt_chain.items(): 
        opt1=val[:3] ## extract information pertaining to options on asset assigned 0.67 weight. -> posn,price,strike
        opt2=val[3:] ## extract information pertaining to options on asset assigned 0.33 weight. ->posn,price,strike
        
        if(opt1[2]>u67 and opt1[0]>0): ## check if long put is ITM at expiry for 0.67 asset
            PnL+=(opt1[2]-u67)*abs(opt1[0])/100 ##Exercise put, payoff is K-S
        if(opt1[0]<0 and opt1[2]>u67): ## check if short put is ITM at expiry for 0.67 asset
            PnL-=(opt1[2]-u67)*abs(opt1[0])/100 ##PnL reduces since put is exercised
        
        if(opt2[2]>u33 and opt2[0]>0): ## check if long put is ITM at expiry for 0.33 asset
            PnL+=(opt2[2]-u33)*abs(opt2[0])/100 ##Exercise put, payoff is K-S
        if(opt2[0]<0 and opt2[2]>u33): ## check if short put is ITM at expiry for 0.33 asset
            PnL-=(opt2[2]-u33)*abs(opt2[0])/100 ##PnL reduces since put is exercised
        
    opt_chain.clear() ## clear options chain after calculating PnL
    return PnL
        
def calc(start,end,col_67,col_33,df,init_cap):
    global last_eq_return
    global opt_chain
    
    dt1=pd.Timestamp(start+timedelta(days=1)).normalize() ## Assume we calculate qtrly returns till day before start
    dt2=(pd.Timestamp(end).normalize())  
    df=df.loc[(df['Date']>=dt1) & (df['Date']<=dt2)] ##filter dataframe between start and end dates for convenience
    df.reset_index(inplace=True,drop=True) 
    
    delta_1=((init_cap*0.67)/df[f'open_{col_67}'][0])/100 ##delta of long asset1(0.67) position
    delta_2=((init_cap*0.33)/df[f'open_{col_33}'][0])/100 ##delta of long asset2(0.33) position
         
    df['opt_PnL']=[0]*df.shape[0] ##init option PnL
    df['opt_posn_1']=[0]*df.shape[0] ##init option1 no of contracts
    df['opt_posn_2']=[0]*df.shape[0] ##init option2 no of contracts
    
    df['opt_posn_1'][0]=(delta_1/abs(df[f'delta_{col_67}'][0])) ##no of put contracts for delta=0 for 0.67 asset
    df['opt_posn_2'][0]=(delta_2/abs(df[f'delta_{col_33}'][0])) ##no of put contracts for delta=0 for 0.33 asset
    df['opt_PnL'][0]=-(df['opt_posn_1'][0]*df[f'put_price_{col_67}'][0]+df['opt_posn_2'][0]*df[f'put_price_{col_33}'][0])
    ##initial PnL for the two puts on 0.67 and 0.33 asset
    
    df[f'PnL_{col_67}'][0]=(1+df[f'Returns_{col_67}'][0])*0.67*init_cap ##initial PnL for long posn in 0.67 asset
    df[f'PnL_{col_33}'][0]=(1+df[f'Returns_{col_33}'][0])*0.33*init_cap ##initial PnL for long posn in 0.67 asset
    
    
    opt_chain[0]=[df['opt_posn_1'][0],df[f'put_price_{col_67}'][0],df[f'open_{col_67}'][0],
                 df['opt_posn_2'][0],df[f'put_price_{col_33}'][0],df[f'open_{col_33}'][0],] ## store posn,price,strike for option positions that have been entered into for the purpose of delta hedging
    
         
    day=1 ## Assuming options have weekly expiry, start at day=1 for index=0 above
    for i in range(1,df.shape[0]): ##iterate over remaining days in the qtr
        
        df[f'PnL_{col_67}'][i]=(1+df[f'Returns_{col_67}'][i])*df[f'PnL_{col_67}'][i-1] ##value of long 0.67 asset posn
        df[f'PnL_{col_33}'][i]=(1+df[f'Returns_{col_33}'][i])*df[f'PnL_{col_33}'][i-1] ##value of long 0.33 asset posn

        day+=1 ##increment no. of days by 1
        
        ## IF at the end of the quarter settle all option positions and find value of invested init capital
        if(i==df.shape[0]-1):
            df['opt_PnL'][i]+=settle_opt(df[f'open_{col_67}'][i],df[f'open_{col_33}'][i])
            last_eq_return.append(init_cap*(((1+df[f'Returns_{col_67}']).prod())*0.67 + ((1+df[f'Returns_{col_33}']).prod())*0.33))
            
        elif(day%5==0): ##If we reach end of the week, settle options positions and rollover the delta hedge
            day=1
            df['opt_PnL'][i]+=settle_opt(df[f'open_{col_67}'][i],df[f'open_{col_33}'][i])
            df['opt_posn_1'][i]=(delta_1/abs(df[f'delta_{col_67}'][i]))
            df['opt_posn_2'][i]=(delta_2/abs(df[f'delta_{col_33}'][i]))
            df['opt_PnL'][i]+=-(df['opt_posn_1'][i]*df[f'put_price_{col_67}'][i]+df['opt_posn_2'][i]*df[f'put_price_{col_33}'][i]) 
            opt_chain[day]=[df['opt_posn_1'][i],df[f'put_price_{col_67}'][i],df[f'open_{col_67}'][i],
                 df['opt_posn_2'][i],df[f'put_price_{col_33}'][i],df[f'open_{col_33}'][i],] ##posn,price,strike
            
        else: ##Adjust delta of your initial options position based on the new delta of the put option to remain delta neutral
            delta_p1=(delta_1-df['opt_posn_1'][i-1]*abs(df[f'delta_{col_67}'][i]))
            delta_p2=(delta_2-df['opt_posn_2'][i-1]*abs(df[f'delta_{col_33}'][i]))
            df['opt_posn_1'][i]=delta_p1/abs(df[f'delta_{col_67}'][i])
            df['opt_posn_2'][i]=delta_p2/abs(df[f'delta_{col_33}'][i])
            df['opt_PnL'][i]=-(df['opt_posn_1'][i]*df[f'put_price_{col_67}'][i]+df['opt_posn_2'][i]*df[f'put_price_{col_33}'][i])
            key=0 if 0 in opt_chain else 1
            opt_chain[day]=[df['opt_posn_1'][i],df[f'put_price_{col_67}'][i],opt_chain[key][2],
                 df['opt_posn_2'][i],df[f'put_price_{col_33}'][i],opt_chain[key][5]]

    return df
df_res=df.copy()
df_res['opt_PnL'] = 0.0 #initialise option PnL

df_res['PnL_nifty']=0.0 #initialise nifty PnL
df_res['PnL_ftse']=0.0 #initialise ftse PnL
df_res['PnL_gold']=0.0 #initialise gold PnL

dates=[] ## to match dates when analyzing performance

for i in range(len(result) - 1): #iterate over quarters

    start = result.loc[i, 'Date'] #qtr starting date
    end = result.loc[i + 1, 'Date'] #qtr end date
    col_67 = res.loc[i, 'col_067'] #name of asset to assign 0.67 to
    col_33 = res.loc[i, 'col_033'] #name of asset to assign 0.33 to
    
    ##initial capital allocation is 1 if we are the beginning of the 3 year product cycle else 
    ## capital allocation is whatever capital was left after investing last quarter
    init_cap=1 if i%12==0 else last_eq_return[-1] 
    
    df_result = calc(start, end, col_67, col_33, df_res,init_cap) ##get result dataframe after being invested
    ##for 1 qtr
    
    ###This is to match dates from the dataframe that is returned and then append those values
    ### to our main dataframe
    mask = (df_res['Date'] > pd.Timestamp(start)) & (df_res['Date'] <= pd.Timestamp(end))
    start_date = df_res.loc[mask, 'Date'].iloc[0]  # First date where mask is True
    end_date = df_res.loc[mask, 'Date'].iloc[-1] 
    dates.append([start_date,end_date])
    if mask.sum() == df_result.shape[0]:
        df_res.loc[mask, 'opt_PnL'] = df_result['opt_PnL'].values
        df_res.loc[mask, 'PnL_nifty'] = df_result['PnL_nifty'].values
        df_res.loc[mask, 'PnL_ftse'] = df_result['PnL_ftse'].values
        df_res.loc[mask, 'PnL_gold'] = df_result['PnL_gold'].values
    else:
        matching_length = min(mask.sum(), df_result.shape[0])
        df_res.loc[mask, 'opt_PnL'].iloc[:matching_length] = df_result['opt_PnL'].iloc[:matching_length].values
        df_res.loc[mask, 'PnL_nifty'].iloc[:matching_length] = df_result['PnL_nifty'].iloc[:matching_length].values
        df_res.loc[mask, 'PnL_ftse'].iloc[:matching_length] = df_result['PnL_ftse'].iloc[:matching_length].values
        df_res.loc[mask, 'PnL_gold'].iloc[:matching_length] = df_result['PnL_gold'].iloc[:matching_length].values
    
df_res=df_res.loc[(df_res['opt_PnL']!=0.0)]
df_res.reset_index(inplace=True,drop=True)
df_res['total'] = df_res[['PnL_nifty', 'PnL_ftse', 'PnL_gold','opt_PnL']].sum(axis=1) ##Find total return on each day
df_res.dropna(inplace=True)
df_res.set_index('Date',inplace=True)

output_dir = "Results"
os.makedirs(output_dir, exist_ok=True)

def save_analysis_to_html(file_name, start_date, end_date, sharpe, cum_return, cagr, sortino, vol, mdd, prices):
    with open(file_name, 'w') as f:
        f.write(f"<html><body>")
        f.write(f"<h2>Analysis Report</h2>")
        f.write(f"<p><b>Start Date:</b> {start_date}</p>")
        f.write(f"<p><b>End Date:</b> {end_date}</p>")
        f.write(f"<p><b>Sharpe Ratio:</b> {sharpe:.2f}</p>")
        f.write(f"<p><b>Cumulative Return:</b> {cum_return:.2f}%</p>")
        f.write(f"<p><b>CAGR:</b> {cagr:.2f}%</p>")
        f.write(f"<p><b>Sortino Ratio:</b> {sortino:.2f}</p>")
        f.write(f"<p><b>Annualised Volatility:</b> {vol:.2f}%</p>")
        f.write(f"<p><b>Maximum Drawdown:</b> {mdd:.2f}%</p>")
        f.write(f"<h3>Performance Chart</h3>")

        # Create the plot and convert it to a base64 string
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(prices / prices[0])
        ax.set_title('Cumulative Performance')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Value')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close(fig)
        img_buffer.seek(0)

        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        f.write(f'<img src="data:image/png;base64,{img_base64}" alt="Performance Chart">')

        f.write(f"</body></html>")

def analyze_and_save(returns, rf, prices, start_date, end_date):
    def sharpe(returns, rf):
        rf_daily = math.pow(1 + rf, 1 / 252) - 1
        excess_returns = returns - rf_daily
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    def cumret(returns):
        return ((1 + returns).prod() - 1) * 100

    def cagr(returns):
        return (math.pow((1 + returns).prod(), 252 / len(returns)) - 1) * 100

    def sortino(returns, rf):
        rf_daily = math.pow(1 + rf, 1 / 252) - 1
        excess_returns = returns - rf_daily
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std()
        return (excess_returns.mean() / downside_deviation) * np.sqrt(252)

    def vol(returns): 
        return returns.std() * np.sqrt(252) * 100

    def MDD(price_series):
        cumulative_return = (1 + price_series.pct_change().fillna(0)).cumprod()
        peak = cumulative_return.cummax()
        drawdown = (cumulative_return - peak) / peak
        return drawdown.min() * 100
    
    # Calculate metrics
    s1 = sharpe(returns, rf)
    s2 = cumret(returns)
    s3 = cagr(returns)
    s4 = sortino(returns, rf)
    s5 = vol(returns)
    s6 = MDD(prices)
    
    # Generate unique file name based on dates
    file_name = os.path.join(output_dir, f"analysis_{start_date}_{end_date}.html")
    save_analysis_to_html(file_name, start_date, end_date, s1, s2, s3, s4, s5, s6, prices)

# Loop to create HTML files for each time period
for i in range(0, len(dates), 12):
    start = pd.Timestamp(dates[i][0]).normalize()
    end = pd.Timestamp(dates[min(i + 11, len(dates) - 1)][1]).normalize()
    analyze_and_save((df_res['total'][start:end]).pct_change().dropna(), 0.05, df_res['total'][start + timedelta(days=1):end], start.date(), end.date())





