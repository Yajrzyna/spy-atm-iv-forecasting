# Short-Horizon Forecasting of ATM Implied Volatility on SPY Options

MSc thesis code (SGH Warsaw / University of Florence, 2026). Can ML models forecast intraday at-the-money implied volatility better than a random walk?

## Setup
- **Data:** OptionsDX SPY 30-minute option chains, 2017–2023. 94.8M quotes filtered to 54.8k ATM rows with 1–7 days to expiry.
- **Split:** train 2017–21, validation 2022, test 2023 (chronological).
- **Models:** OLS, MLP, LSTM and Ridge stacking, each benchmarked against a random walk.
- **Metric:** out-of-sample R² vs the random walk, plus direction hit rate.

## Results (test 2023)
- **OLS:** R² 0.192, hit rate 66%
- **Stacking:** R² 0.177, hit rate 67%
- **LSTM:** R² 0.172, hit rate 68%
- **MLP:** R² 0.060, hit rate 64%

All models beat the random walk, and simple OLS is best. Skill is strongest at 1–2 days to expiry and weakest when VIX is low. All tables are in `results/evaluation_report.txt`.

## Corrections after submission
- **Vega units:** the trading measure multiplied vega (per vol point) by an IV change in decimals, so P&L was 100× too small. I caught it with a perfect-foresight test (`test_metrics.py`).
- **LSTM patience:** the code used 30 instead of the validated 20.

This repo contains the corrected code. With the units fixed, the vega-only Sharpe is positive in most cells. It ignores theta, hedging and exit costs, so it is **not** evidence of a tradable strategy.

## Code
- `code/1_feature_engineering/`: 01 raw → parquet, 02 filter, 03 features, 04 split
- `code/2_models/05_train_and_evaluate.py`: trains and evaluates all models
- `code/2_models/06–08_figure_*.py`: figures in `results/figures/`
- `code/2_models/metrics.py`: metrics, tested in `test_metrics.py`

## Run
The data is licensed and not included (see `data/README.md`).

```bash
pip install -r requirements.txt
export OPTIONSDX_DIR="/path/to/OptionsDX Data"
```

Then run the scripts in numeric order. The tests and figures need no data:

```bash
cd code/2_models
python -m unittest test_metrics
python 07_figure_vega_sharpe.py
```

MIT licence.
