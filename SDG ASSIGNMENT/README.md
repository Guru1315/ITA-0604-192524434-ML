# SDG assignment: How machine learning predicts employment trends

A data analysis project linked to **UN Sustainable Development Goal 8: Decent Work and Economic Growth**. It cleans a sector-level employment dataset, analyses it with NumPy and Pandas, and tests whether machine learning models can predict next-year employment growth.

> **The data is simulated, not real.** The dataset is a made-up 16-year, 10-sector panel built to follow widely reported patterns (growth in IT, health and green jobs, decline in agriculture and manufacturing, rising automation, and the 2020 shock). All figures are illustrative and are **not official statistics**. The report says the same on its first page. The method is what would be used on real labour-force data.

## Files

| File | What it is |
|---|---|
| `ML_Employment_Trends_Report.pdf` | The 9-page written report |
| `analysis.py` | The script that builds the data, cleans it, trains the models and draws the charts |
| `employment_raw.csv` | The generated data before cleaning (163 rows) |
| `employment_clean.csv` | The data after cleaning (160 rows: 10 sectors times 16 years, 2010 to 2025) |

Columns: `year, sector, employment_m` (employment in millions), `gdp_growth, automation_index, digital_skill_demand, job_postings_index, wage_growth`.

## What the script does

1. **Builds the data.** It simulates 10 sectors (IT and Software, Healthcare, Renewable Energy, E-commerce and Logistics, Education, Financial Services, Construction, Manufacturing, Agriculture, Retail) and then deliberately adds problems so the cleaning step has real work to do: 10 missing values, 3 duplicate rows and 2 entry errors that are ten times too large. The random generator is seeded (`default_rng(42)`), so the data is repeatable.
2. **Cleans it.** It removes the duplicates, treats any employment value more than 5 times its sector's median as an error, and fills the gaps by interpolating within each sector.
3. **Analyses it** with NumPy and Pandas: sector summaries and growth rates.
4. **Predicts next-year growth.** Each row uses the previous year's values as inputs. The models are trained on 2012 to 2021 (100 rows) and tested on 2022 to 2025 (40 rows), so they are judged on years they never saw.
5. **Forecasts 2026 to 2030** for each sector with a straight-line trend fitted to the logarithm of employment.
6. **Draws seven charts** (workflow, sector trends, growth rates, correlations, model comparison, feature importance and the forecast).

## Results

| Model | MAE (percentage points) | R-squared |
|---|---|---|
| Naive (last year's growth) | 0.54 | 0.95 |
| Linear Regression (Ridge) | 1.81 | 0.68 |
| Random Forest | 0.74 | 0.91 |
| Gradient Boosting | 1.00 | 0.87 |

The Random Forest was the best of the machine learning models, but the simple "repeat last year's growth" baseline was better still on this small dataset. The report says why: growth is very persistent, so there is little for a complex model to add with only 100 training rows. These numbers are from the report and describe the simulated data only.

## Run it

```bash
pip install numpy pandas scikit-learn matplotlib
```

Before running `analysis.py`, change this line near the top, because it points at a folder from the machine where the script was written:

```python
OUT = "/home/claude/work/"
```

Set `OUT` to a folder that exists on your computer (ending in `/`) and create a `charts` folder inside it. The script does not create folders, so it will fail without them. It then writes `employment_raw.csv`, `employment_clean.csv`, `sector_summary.csv`, `model_results.csv`, `metrics.json` and the seven charts there. Only the two CSV files are stored in this repository.
