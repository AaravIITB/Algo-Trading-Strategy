# Algo-Trading-Strategy
A quantitative backtesting framework for Nifty50 stocks with portfolio allocation, technical indicators, and performance metrics (CAGR, Sharpe, Max Drawdown).
## Nifty Quant Backtester
A Python-based quantitative backtesting framework for Nifty50 stocks with portfolio allocation, technical indicators, and performance metrics.

This project simulates a balanced hybrid trading strategy on a portfolio of Nifty50 companies, using strict entry/exit rules and risk management. It outputs equity curves, portfolio performance, and metrics like CAGR, Sharpe ratio, volatility, and maximum drawdown.

## Features
### Technical Indicators
1. Moving Averages (MA20, MA50)
2. RSI (14-day)
3. ATR (14-day)
4. ADX / DMI (14-day)
5. Bollinger Band Width
6. EMA10

### Strategy Logic
1. Entry based on MA crossovers, RSI > 50, volume confirmation, and trend filters
2. Trailing ATR-based stop-loss
3. Profit targets (+3 ATR)
4. Fast exits when trend weakens

### Portfolio Allocation
The portfolio is constructed from 10 carefully chosen companies within the Nifty50 index. Selection was guided by a combination of sector leadership, diversification, liquidity, and historical consistency, rather than short-term price returns.

Selection Criteria:
1. Market Leadership – Companies that are leaders or strong contenders in their industries (e.g., Bajaj Finance in NBFCs, TCS in IT, Titan in retail).
2. Sectoral Diversification – Exposure across financial services, IT, FMCG, energy, and consumer goods to balance growth with stability.
3. Liquidity & Stability – Only highly liquid Nifty50 stocks were included, ensuring that backtested strategies are tradable in real markets.
4. Historical Consistency – Preference for firms with strong earnings growth, stable fundamentals, and resilient price behavior.
5. Risk-Adjusted Exposure – Weights were assigned to balance high-growth stocks (Bajaj Finance, Adani Enterprises) with more defensive names (NTPC, Hindustan Unilever).

### Performance Metrics
1. CAGR (Compounded Annual Growth Rate)
2. Sharpe Ratio
3. Annualized Volatility
4. Maximum Drawdown
5. Total Returns

### Results
1. Equity Curve of combined portfolio
2. Performance metrics csv
3. Buy and Sell points plotted on graph and downloaded all the points in a csv of each company

## *Backtesting done from date 2018-01-01 to 2019-06-28

## Next Step to be taken
This strategy has been applied on 10 companies of NIFTY 50 Index.

I have created a screener query :
1. Market Capitalization > 1000 AND
2. Debt to Equity < 1 AND
3. Interest Coverage Ratio > 3 AND
4. Current ratio > 1.5 AND
5. Sales growth 5Years > 10 AND
6. Profit growth 5Years > 10 AND
7. Return on capital employed > 15 AND
8. Return on equity > 15 AND
9. Free cash flow 3years  > 0 AND
10. Price to Earning < Industry PE  AND
11. Change in FII holding 3Years >0 

Through this I'll filter out companies and apply my strategy on those companies. I aim to a better distribution of weights depending on the market in the next step.
After successful backtesting on these companies. I aim to deploy my strategy in the current market to automate my trading.
