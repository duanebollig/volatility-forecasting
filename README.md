# Volatility Forecasting
Comparing naive, HAR-RV, LightGBM next-week realized variance forecasts for SPY data (2000-2026), validated using walk-forward.

## Why
In the market, the direction of returns is *close to unpredictable*. Market variance however, can be more accurately predicted. This is because calm weeks tend to follow calm weeks, and wild follow wild. This is called **volatility clustering**, and traders use this to *size their positions*, controlling the amount of risk on their position by shrinking its size when forecast volatility is high.

## Data
Daily SPY closes from 2000-01-01 to 2026-10-28, from yfinance. Closes are dividend- and split- adjusted so returns reflect real price changes, not fake drops on dividend days.

- 6,746 trading days downloaded

- 6,719 rows after building features and target

- 5,315 out-of-sample forecasts (first 1,400 rows train, 4-row gap to prevent look-ahead leakage)

`data/` isn't committed, so run `download.py` first. Re-downloading later adds newer days, so numbers shift slightly.

## Target
The target for day t is the realized variance over the next 5 trading days, measured as the average of squared daily log returns, expressed as:

$$y_t = \frac{1}{5}\sum_{i=1}^{5}r_{t+i}^2 , \qquad r_t = \ln\frac{P_t}{P_{t-1}}$$

SPY drifts up very slowly (about 0.03% a day) but bounces ~1% a day, so the drift can be ignored and a day's squared log return can be used as an *unbiased but noisy* estimate of that day's variance. Since log returns are time-additive and nearly uncorrelated, a period's variance is the sum of its daily variances: 

$$\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X,Y) \approx \mathrm{Var}(X) + \mathrm{Var}(Y) \quad \text{since } \mathrm{Cov}(X,Y) \approx 0$$

For this project, I used a 5-day period to represent a trading week's variance, averaging the 5 readings to express weekly variance as an average daily variance, matching the features' units. Volatilities don't add, so the model works in variance; volatility is only used for display.

## Features
On any given day t, the models see the same three inputs: average squared log returns over a daily, weekly, and monthly lookback window (1, 5, and 22 trading days respectively).

- **daily**: today's squared log return,

  $$\text{daily}_t = r_t^2$$

- **weekly**: the average of the last 5 squared log returns,

  $$\text{weekly}_t = \frac{1}{5}\sum_{i=0}^{4}r_{t-i}^2$$

- **monthly**: the average of the last 22 squared log returns,

  $$\text{monthly}_t = \frac{1}{22}\sum_{i=0}^{21}r_{t-i}^2$$

These windows come from the HAR model (Corsi, 2009).[^corsi] Corsi argues that volatility holds important information at each scale because traders think on different horizons, so volatility changes fast on the daily, slow on the monthly, and intermediate on the weekly. Together, they let the forecast react quickly to shocks without forgetting the longer regime. For example:

1. **A huge day after a calm month**: daily = high, weekly = medium, monthly = low. So, HAR's forecast rises, but not up to the spike.

2. **A calm day after a chaotic month**: daily = low, weekly = medium, monthly = high. So, HAR's forecast lowers, but not to the calm day's level.

To prevent leakage, features use day t and earlier, while the target starts at t+1.

## References
[^corsi]: Corsi, F. (2009). "A Simple Approximate Long-Memory Model of Realized Volatility." *Journal of Financial Econometrics*, 7(2), 174–196.
