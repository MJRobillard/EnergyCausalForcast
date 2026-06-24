# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup
```bash
pip install -r requirements.txt
sudo apt-get install graphviz          # required for SCM diagram rendering
```

### WAUE track (original paper replication)
```bash
python reproduce.py                    # fetch data + train
python reproduce.py --skip-fetch       # train on existing local data
python model/train.py                  # train/test eval + 5-fold CV
python model/render_diagram.py --out results/scm_model.pdf
python model/render_diagram.py --generative --out results/scm_model_generative.pdf
```

### California sectoral pipeline
```bash
python model/train_california.py                          # train on CAISO data (slow, ~5 min)
python -m pipeline.run                                    # full 8-stage pipeline
python -m pipeline.run --scenario LocalRel --no-plots     # skip figure generation
python -m pipeline.run --validate-only                    # run validation checks only
```

### Data fetching
```bash
EIA_API_KEY=your_key python data/fetch_load.py     # WAUE demand (EIA key required)
python data/fetch_weather.py                        # ERA5/Open-Meteo weather (no key)
python data/fetch_historical.py                     # California historical data (no key)
```

`EIA_API_KEY` is only required for the WAUE EIA fetch. All California data uses Open-Meteo (free, no key).

---

## Architecture

### Two independent research tracks

Both tracks share `model/scm.py` but are otherwise separate:

- **WAUE track**: `data/fetch_*.py` → `model/train.py` / `reproduce.py` — replicates Ma et al. 2024 (arXiv:2512.11653) on the Western US balancing authority.
- **California sectoral track**: `model/train_california.py` → `pipeline/` → `notebooks/` — active research applying the SCM to CEC 2025–2050 electrification scenarios and heatwave stress analysis.

### California pipeline — 3-stage design

The pipeline is not a single model. It has three distinct stages that must be understood together:

**Stage 1 — Pyro SVI training** (`model/train_california.py`):
Trains on historical CAISO hourly demand (2023–2025) using Stochastic Variational Inference with an `AutoNormal` guide. Outputs posterior means to `results/california/param_store.pt` and metrics to `results/california/summary.json`. Key posteriors used downstream: `k_cool`, `k_max`, `k_cool_sq`.

**Stage 2 — Calibration layer** (`pipeline/scm/calibration.py`):
Sector betas (β_RES, β_COM, β_IND) are **not** Pyro posteriors. They are computed algebraically by decomposing the observed 2022 heatwave uplift anchor (4,625 MW at Sep 5 18:00):

```
OBSERVED_2022_UPLIFT_MW = HVAC_structural + d_aafs + d_ev + d_dc + d_residual
```

`d_residual` absorbs multi-day thermal lag, behavioral responses, and unexplained demand. It is the largest term at forward years (~69% of δMNL in worst-case 2035). It scales forward proportionally to fleet growth × persistence ratio — it is a calibration residual, not a structurally derived quantity.

**Stage 3 — Forward scenarios** (`pipeline/scm/heatwave_uplift.py`):
Three uplift methods: `STATIC_2022` (flat 4,625 MW benchmark), `FLEET_SCALED` (legacy), `YEAR_NATIVE` (severity-matched weather applied to year-Y climatology and fleet — the method used in all current notebooks). The year-native method applies the 2022 ΔT anomaly on top of each projection year's climatological normal, so the absolute temperatures change while the severity is held constant.

### Key constants are pre-baked — no training required to run notebooks

`pipeline/config.py` contains the posterior means already extracted from training:
- `K_COOL = 62.030` MW/°F, `K_MAX = 53.692` MW/°F, `K_COOL_SQ = -5.110` MW/°F²
- `T_COOL_BASE = 64.360°F`, `T_MAX_BASE = 85.662°F`

The full pipeline and all notebooks run from these constants without re-training. `results/california/summary.json` is optional (pipeline falls back to config constants if absent).

### Validation Check 1 is a hard stop

`pipeline/validation.py` runs 4 checks after calibration. **Check 1** (MW conservation) halts the pipeline if HVAC + fleet + DC + residual don't sum to 4,625 MW ±1 MW. This fires whenever `K_COOL`, sector shares, or CEC data change. Checks 2–4 warn only.

### Data files are gitignored — exact paths in config

CEC scenario Excel files live in `data/CEC/` and weather parquets in `data/california/`. Neither is in git. The exact filenames (including CEC timestamp prefixes) are defined in `pipeline/config.py: SCENARIO_FILES`. Fetch missing California data with `python data/fetch_historical.py`.

### Hub-composite temperature

`pipeline/data/weather.py` computes a load-weighted composite from 4 region parquets: Bay Area (65%), LA (15%), Riverside (15%), San Diego (5%). Riverside is also loaded as a separate standalone series — it serves as the inland heat extreme proxy (`T_rv`) for `k_max` and DC thermal calculations. Climatological normals exclude 2022 (the event year).

### Notebook dependency split

- `heat_wave_policy_scenario_analysis.ipynb` and `heat_wave_sector_causal*.ipynb` import heavily from `pipeline/` submodules and require all CEC + weather data files to be present.
- `scm_counterfactual_cec.ipynb` is largely self-contained — SCM parameters are hardcoded inline and `import torch; import pyro` are vestigial (no inference runs).

### Counterfactual engine

`pipeline/counterfactuals/engine.py` applies policy overrides via `CounterfactualSpec` dataclass fields — scalar multipliers on `cal` fields, per-year lambda functions for fleet adjustments, sector beta overrides. New counterfactuals are defined in `pipeline/counterfactuals/scenarios.py` by composing these overrides. The engine recomputes the full trajectory for each spec and appends a `vs_baseline` column.
