# EnergyForecast

Structural causal modeling experiments for hourly electricity demand forecasting, based on the WAUE case study from *Causal Inference in Energy Demand Prediction*.

The project fetches public load and weather data, builds a Pyro structural causal model (SCM), trains it with stochastic variational inference, and evaluates demand forecasts with MAPE.

## Project Layout

```text
data/
  fetch_load.py       # Fetch WAUE hourly demand from the EIA API
  fetch_weather.py    # Fetch ERA5 weather from Open-Meteo
  raw/                # Generated local CSVs, ignored by git
model/
  scm.py              # Pyro SCM, training helper, prediction helper
  train.py            # Train/test evaluation and cross-validation
  render_diagram.py   # Render the Pyro model graph
results/              # Generated outputs, ignored by git
reproduce.py          # End-to-end data fetch + training pipeline
requirements.txt      # Python dependencies
```

## Data Sources

- Electricity demand: EIA API v2, WAUE balancing authority.
- Weather: Open-Meteo Historical Weather API / ERA5 reanalysis.
- Period: September 2023 through August 2025.
- Representative weather coordinate: `44.6321, -100.2753`, matching the paper.

The EIA load fetch requires an API key:

```bash
export EIA_API_KEY=your_key_here
```

Do not commit API keys. Keep them in `.env` or your shell environment.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use conda instead:

```bash
conda create -n energyforecast python=3.11
conda activate energyforecast
pip install -r requirements.txt
```

Graph rendering also requires the Graphviz system package:

```bash
sudo apt-get install graphviz
```

## Reproduce

Fetch data and train:

```bash
python reproduce.py
```

Use existing local CSVs:

```bash
python reproduce.py --skip-fetch
```

Train only:

```bash
python model/train.py
```

Render the SCM diagram:

```bash
python model/render_diagram.py --out results/scm_model.pdf
```

Render the fully generative graph, without conditioning on observed weather/demand:

```bash
python model/render_diagram.py --generative --out results/scm_model_generative.pdf
```

## Model Notes

The SCM decomposes total demand into:

- HVAC demand from temperature, humidity, and wind.
- Daily activity demand from Fourier features over hour of day.
- Yearly/seasonal demand from Fourier features over month.
- Lighting demand from solar radiation during active hours.

The training code uses Pyro SVI with an `AutoNormal` guide and Adam optimizer. Predictions use the posterior mean of deterministic `E_mu`, with observed weather held fixed and demand left unobserved.

The paper reports:

- Train MAPE: `3.23%`
- Test MAPE: `3.84%`
- 5-fold CV MAPE: `3.88%`

Local results depend on the exact SCM specification and prior choices in `model/scm.py`. Generated metrics are written to `results/summary.json`, which is intentionally ignored by git.

## Git Hygiene

The repository is set up to track source code and documentation only. The following are intentionally ignored:

- `.env` and other secret files.
- Downloaded raw data under `data/raw/`.
- Generated results under `results/`.
- Local caches such as `__pycache__`, `.pytest_cache`, and virtual environments.
- Cursor/specstory workspace metadata.

Before pushing, check:

```bash
git status
git diff
```

Only commit files you intend to publish.
