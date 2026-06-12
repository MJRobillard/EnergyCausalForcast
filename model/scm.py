
"""
Structural Causal Model for hourly electricity demand (WAUE).
Reproduces the method from Ma et al. 2024 (arXiv:2512.11653).

DAG: Hour, Month -> Weather variables -> Demand components -> Total demand
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# --- Constants from paper ---
T_RH = 70.0        # °F, humidity effect threshold
T_W1 = 50.0        # °F, wind cold threshold (below: heating effect)
T_W2 = 80.0        # °F, wind hot threshold (above: cooling effect)
N_HARMONICS_TEMP_YEARLY = 3
N_HARMONICS_TEMP_DAILY = 3
N_HARMONICS_DAILY = 4   # for E_daily
N_HARMONICS_YEARLY = 3  # for E_yearly
LIGHTING_START = 5       # 5 AM
LIGHTING_END = 24        # midnight (exclusive)

# Default (WAUE) priors — override via make_model(priors={...})
DEFAULT_PRIORS: dict = {
    "E0":           (3485.0, 40.0),
    "k_cool":       (30.0,   40.0),   # cooling sensitivity (T > T_cool_base)
    "k_heat":       (10.0,   40.0),   # heating sensitivity (T < T_heat_base)
    "T_cool_base":  (65.0,    4.0),   # °F, cooling onset
    "T_heat_base":  (55.0,    4.0),   # °F, heating onset
    "T_base":       (47.0,    4.0),   # for temperature generative model
    "a":            ([-500., 360., 40., 0., 0., 0., 0., 0.], 40.0),
    "alpha":        ([-50.,  100., 50., 50., 0., 0.],        40.0),
    "mu_rh_base":   (63.0,  20.0),
    "mu_w_base":    (16.0,  10.0),
    "rad_base":     (100.0, 50.0),
}


def make_tensors(df: pd.DataFrame) -> dict:
    """Convert preprocessed DataFrame to torch tensors."""
    t = lambda col: torch.tensor(df[col].values, dtype=torch.float32)
    out = dict(
        h=t("hour"), m=t("month"),
        T=t("temperature_f"), RH=t("humidity_pct"),
        W=t("wind_mph"), Rad=t("solar_radiation_wm2"),
        E_obs=t("demand_mwh"),
    )
    # Optional extended features — present when train_california builds them
    for col, key in [("dow", "dow"), ("holiday", "holiday"),
                     ("E_lag24", "E_lag24"), ("E_lag168", "E_lag168")]:
        if col in df.columns:
            out[key] = t(col)
    return out


def _fourier(x: torch.Tensor, period: float, n: int) -> torch.Tensor:
    """Return 2n-length vector of sin/cos harmonics for each element of x."""
    terms = []
    for j in range(1, n + 1):
        terms.append(torch.sin(2 * np.pi * j * x / period))
        terms.append(torch.cos(2 * np.pi * j * x / period))
    return torch.stack(terms, dim=-1)  # (N, 2n)


def make_model(priors: dict | None = None):
    """
    Return a Pyro model function with the given prior means.
    Unspecified keys fall back to DEFAULT_PRIORS.
    Use this to swap in data-driven priors for a new region.
    """
    p = {**DEFAULT_PRIORS, **(priors or {})}

    def _model(h, m, T=None, RH=None, W=None, Rad=None, E_obs=None,
               dow=None, holiday=None, E_lag24=None, E_lag168=None):
        """
        Full generative model.  When T/RH/W/Rad are supplied the weather
        likelihood terms are conditioned on observed values.
        When E_obs is supplied the demand likelihood is conditioned.
        """
        N = h.shape[0]

        # ------------------------------------------------------------------ #
        # Temperature model (Eq. 6)                                           #
        # ------------------------------------------------------------------ #
        Fm_temp = _fourier(m, 12.0, N_HARMONICS_TEMP_YEARLY)  # (N, 6)
        Fh_temp = _fourier(h, 24.0, N_HARMONICS_TEMP_DAILY)   # (N, 6)

        c = pyro.sample("c", dist.Normal(
            torch.tensor([-4.6, 6.4, -1.6, -0.86, 0.0, 0.0], dtype=torch.float32),
            torch.ones(6) * 4.0
        ).to_event(1))
        d = pyro.sample("d", dist.Normal(
            torch.tensor([-17.0, -22.0, -2.3, -2.6, 0.0, 0.0], dtype=torch.float32),
            torch.ones(6) * 10.0
        ).to_event(1))
        T_base_mu, T_base_sig = p["T_base"]
        T_base = pyro.sample("T_base", dist.Normal(T_base_mu, T_base_sig))
        sigma_temp = pyro.sample("sigma_temp", dist.LogNormal(2.0, 0.5))

        T_mu = (Fm_temp @ c) + (Fh_temp @ d) + T_base
        with pyro.plate("obs_temp", N):
            T_samp = pyro.sample("T_obs", dist.Normal(T_mu, sigma_temp), obs=T)

        T_use = T if T is not None else T_samp

        # ------------------------------------------------------------------ #
        # Humidity, Wind, Radiation (Eq. 7)                                   #
        # ------------------------------------------------------------------ #
        Fm_wx = _fourier(m, 12.0, 2)   # (N, 4)

        mu_rh_mu, mu_rh_sig = p["mu_rh_base"]
        mu_rh_base = pyro.sample("mu_rh_base", dist.Normal(mu_rh_mu, mu_rh_sig))
        c_rh = pyro.sample("c_rh", dist.Normal(
            torch.zeros(4), torch.ones(4) * 10.0
        ).to_event(1))
        mu_rh = mu_rh_base + Fm_wx @ c_rh
        sig_rh = pyro.sample("sig_rh", dist.LogNormal(2.5, 0.5))

        mu_w_mu, mu_w_sig = p["mu_w_base"]
        mu_w_base = pyro.sample("mu_w_base", dist.Normal(mu_w_mu, mu_w_sig))
        c_w = pyro.sample("c_w", dist.Normal(
            torch.zeros(4), torch.ones(4) * 5.0
        ).to_event(1))
        mu_w = mu_w_base + Fm_wx @ c_w
        sig_w = pyro.sample("sig_w", dist.LogNormal(2.0, 0.5))

        Fh_rad = _fourier(h, 24.0, 3)  # (N, 6)
        Fm_rad = _fourier(m, 12.0, 2)  # (N, 4)
        rad_mu, rad_sig = p["rad_base"]
        rad_base = pyro.sample("rad_base", dist.Normal(rad_mu, rad_sig))
        a_rad = pyro.sample("a_rad", dist.Normal(
            torch.zeros(6), torch.ones(6) * 80.0
        ).to_event(1))
        b_rad = pyro.sample("b_rad", dist.Normal(
            torch.zeros(4), torch.ones(4) * 40.0
        ).to_event(1))
        mu_rad = (rad_base + Fh_rad @ a_rad + Fm_rad @ b_rad).clamp(min=0.0)
        sig_rad = pyro.sample("sig_rad", dist.LogNormal(5.0, 0.5))

        with pyro.plate("obs_weather", N):
            RH_samp = pyro.sample("RH_obs", dist.Normal(mu_rh, sig_rh), obs=RH)
            W_samp = pyro.sample("W_obs", dist.Normal(mu_w, sig_w), obs=W)
            Rad_samp = pyro.sample("Rad_obs", dist.Normal(mu_rad, sig_rad), obs=Rad)

        RH_use = RH if RH is not None else RH_samp
        W_use = W if W is not None else W_samp
        Rad_use = Rad if Rad is not None else Rad_samp

        # ------------------------------------------------------------------ #
        # Energy demand components                                            #
        # ------------------------------------------------------------------ #

        # Asymmetric HVAC: separate cooling and heating slopes with
        # learnable thresholds, replacing the symmetric V-shape from the paper.
        k_cool_mu, k_cool_sig = p["k_cool"]
        k_heat_mu, k_heat_sig = p["k_heat"]
        T_cool_mu, T_cool_sig = p["T_cool_base"]
        T_heat_mu, T_heat_sig = p["T_heat_base"]
        k_cool     = pyro.sample("k_cool",      dist.Normal(k_cool_mu, k_cool_sig))
        k_heat     = pyro.sample("k_heat",      dist.Normal(k_heat_mu, k_heat_sig))
        T_cool_base = pyro.sample("T_cool_base", dist.Normal(T_cool_mu, T_cool_sig))
        T_heat_base = pyro.sample("T_heat_base", dist.Normal(T_heat_mu, T_heat_sig))
        E0_mu, E0_sig = p["E0"]
        E0 = pyro.sample("E0", dist.Normal(E0_mu, E0_sig))
        cooling = torch.clamp(T_use - T_cool_base, min=0.0)
        heating = torch.clamp(T_heat_base - T_use, min=0.0)
        E_base = k_cool * cooling + k_heat * heating + E0

        # Humidity effect (Eq. 9)
        delta_rh = pyro.sample("delta_rh", dist.Normal(0.0, 5.0))
        mask_rh = (T_use > T_RH).float()
        E_humid = delta_rh * RH_use * mask_rh

        # Wind effect (Eq. 10)
        gamma_w = pyro.sample("gamma_w", dist.Normal(0.0, 5.0))
        lambda_w = pyro.sample("lambda_w", dist.Normal(0.0, 5.0))
        mask_cold = (T_use < T_W1).float()
        mask_hot = (T_use > T_W2).float()
        E_wind = gamma_w * W_use * mask_cold - lambda_w * W_use * mask_hot

        E_hvac = E_base + E_humid + E_wind

        # Daily activity (Eq. 11)
        Fh_act = _fourier(h, 24.0, N_HARMONICS_DAILY)   # (N, 8)
        a_mu, a_sig = p["a"]
        a = pyro.sample("a", dist.Normal(
            torch.tensor(a_mu, dtype=torch.float32),
            torch.ones(8) * a_sig
        ).to_event(1))
        E_daily = Fh_act @ a

        # Yearly cycle (Eq. 12)
        Fm_yr = _fourier(m, 12.0, N_HARMONICS_YEARLY)   # (N, 6)
        alpha_mu, alpha_sig = p["alpha"]
        alpha = pyro.sample("alpha", dist.Normal(
            torch.tensor(alpha_mu, dtype=torch.float32),
            torch.ones(6) * alpha_sig
        ).to_event(1))
        E_yearly = Fm_yr @ alpha

        # Day-of-week effect (optional — only when dow tensor is provided)
        E_dow = torch.zeros(N)
        if dow is not None:
            a_dow = pyro.sample("a_dow", dist.Normal(
                torch.zeros(7), torch.ones(7) * 200.0
            ).to_event(1))
            E_dow = a_dow[dow.long()]

        # Holiday effect (optional)
        E_holiday = torch.zeros(N)
        if holiday is not None:
            delta_holiday = pyro.sample("delta_holiday", dist.Normal(0.0, 500.0))
            E_holiday = delta_holiday * holiday

        # Lagged demand — base predictor; SCM terms model corrections on top.
        # Keeping lags as the level predictor means E0/Fourier terms only need
        # to explain deviations, so their ~0-mean priors are well-calibrated.
        E_lag = torch.zeros(N)
        if E_lag24 is not None and E_lag168 is not None:
            w_lag24  = pyro.sample("w_lag24",  dist.Normal(0.5, 0.2))
            w_lag168 = pyro.sample("w_lag168", dist.Normal(0.3, 0.2))
            E_lag = w_lag24 * E_lag24 + w_lag168 * E_lag168

        # Lighting (Eq. 13)
        L0 = pyro.sample("L0", dist.LogNormal(5.0, 1.0))
        beta = pyro.sample("beta", dist.LogNormal(-5.0, 1.0))
        active_hour = ((h >= LIGHTING_START) & (h < LIGHTING_END)).float()
        E_light = L0 * torch.exp(-beta * Rad_use) * active_hour

        # Total demand
        E_mu = E_hvac + E_light + E_daily + E_yearly + E_dow + E_holiday + E_lag
        pyro.deterministic("E_mu", E_mu, event_dim=1)
        sigma_E = pyro.sample("sigma_E", dist.LogNormal(4.0, 1.0))

        with pyro.plate("obs_demand", N):
            pyro.sample("E_obs", dist.Normal(E_mu, sigma_E), obs=E_obs)

        return E_mu

    return _model


# Default WAUE model — backward-compatible
model = make_model()


def train(tensors: dict, num_steps: int = 5000, lr: float = 0.01,
          seed: int = 42, model=None) -> tuple[SVI, AutoNormal]:
    _model = model if model is not None else globals()["model"]
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    guide = AutoNormal(_model, init_loc_fn=pyro.infer.autoguide.init_to_median)
    optimizer = Adam({"lr": lr})
    svi = SVI(_model, guide, optimizer, loss=Trace_ELBO())

    print(f"Training SCM for {num_steps} steps ...")
    for step in range(1, num_steps + 1):
        loss = svi.step(**tensors)
        if step % 500 == 0:
            print(f"  step {step:5d}  ELBO loss = {loss:.1f}")

    return svi, guide


def predict(guide: AutoNormal, tensors: dict,
            num_samples: int = 200, model=None) -> np.ndarray:
    """Return posterior mean of noiseless E_mu for each timestep."""
    _model = model if model is not None else globals()["model"]
    predictive = pyro.infer.Predictive(_model, guide=guide,
                                       num_samples=num_samples)
    inp = {k: v for k, v in tensors.items() if k != "E_obs"}
    samples = predictive(**inp)
    return samples["E_mu"].mean(0).detach().numpy().flatten()


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def render_diagram(
    filename: str,
    tensors: Optional[dict] = None,
    *,
    render_distributions: bool = True,
    conditioned: bool = True,
) -> None:
    """
    Render the SCM as a Graphviz diagram via pyro.render_model().

    Requires the graphviz system binary (``apt install graphviz`` or
    ``conda install -c conda-forge graphviz``) plus ``pip install graphviz``.

    Args:
        filename: Output path including extension (e.g. ``results/scm.pdf``).
        tensors: Model kwargs; defaults to a small dummy batch.
        render_distributions: Annotate each RV with its prior.
        conditioned: If True (default), include observed weather/demand nodes.
    """
    if tensors is None:
        n = 3
        tensors = dict(
            h=torch.zeros(n),
            m=torch.ones(n),
            T=torch.full((n,), 70.0),
            RH=torch.full((n,), 50.0),
            W=torch.full((n,), 5.0),
            Rad=torch.full((n,), 200.0),
            E_obs=torch.full((n,), 5000.0),
        )
    if not conditioned:
        tensors = {k: v for k, v in tensors.items()
                   if k not in ("T", "RH", "W", "Rad", "E_obs")}

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    pyro.render_model(
        model,
        model_kwargs=tensors,
        filename=filename,
        render_distributions=render_distributions,
    )
