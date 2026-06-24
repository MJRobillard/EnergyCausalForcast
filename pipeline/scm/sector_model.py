"""
Sector-disaggregated SCM demand uplift for a single year.

For each year Y and sector s ∈ {RES, COM, IND}, the causal uplift is:

    δ_s(Y) = β_s × UNADJ_s(Y) × ΔT_hub          (HVAC structural, sector-scaled)
           + AAFS_s(Y) × β_aafs × ΔT_hub          (electrified buildings, by sector)
           + EV_s(Y) × β_ev × ΔT_hub               (EV fleet, by sector)

    δ_DC(Y) = DC(Y) × f_dc_thermal                 (DC thermal, pooled)
    δ_lag   = d_residual                            (fixed lag/behavioral)

    δ_total(Y) = Σ_s δ_s(Y) + δ_DC(Y) + δ_lag

Physical capacity ceiling (Validation 2):
    δ_s ≤ UNADJ_s(Y) × nameplate_cooling_fraction[s]

If clipped, a CapacitySaturationWarning is attached to the row.

Exposes
-------
SectorUpliftRow  dataclass
compute_sector_uplift(yr, peak_row, cal, sector_shares_yr) -> SectorUpliftRow
"""

from __future__ import annotations
from dataclasses import dataclass, field
import warnings

from pipeline.config import AAFS_SECTOR_SPLIT, LIGHT_EV_SECTOR_SPLIT, AATE_LDV_SECTOR_SPLIT
from pipeline.scm.calibration import SCMCalibration


@dataclass
class SectorUpliftRow:
    """Per-year disaggregated uplift result."""
    year:       int
    scenario:   str

    # ── Sector HVAC terms (MW) ────────────────────────────────────────────
    d_hvac_res: float
    d_hvac_com: float
    d_hvac_ind: float

    # ── Sector AAFS terms ─────────────────────────────────────────────────
    d_aafs_res: float
    d_aafs_com: float

    # ── Sector EV terms ───────────────────────────────────────────────────
    d_ev_res:   float
    d_ev_com:   float
    d_ev_ind:   float

    # ── DC thermal ────────────────────────────────────────────────────────
    d_dc:       float

    # ── Fixed residual ────────────────────────────────────────────────────
    d_residual: float

    # ── Totals ────────────────────────────────────────────────────────────
    d_total:    float
    d_static:   float          # naive constant (4,625 MW)
    mnl_cec:    float          # CEC baseline managed net load
    mnl_hw:     float          # MNL + d_total

    # ── Validation flags ──────────────────────────────────────────────────
    capacity_clipped: bool = False
    clipped_sectors:  list[str] = field(default_factory=list)
    clip_amount_mw:   float = 0.0

    @property
    def d_hvac_total(self) -> float:
        return self.d_hvac_res + self.d_hvac_com + self.d_hvac_ind

    @property
    def d_aafs_total(self) -> float:
        return self.d_aafs_res + self.d_aafs_com

    @property
    def d_ev_total(self) -> float:
        return self.d_ev_res + self.d_ev_com + self.d_ev_ind

    def sector_summary(self) -> dict[str, float]:
        """Return total MW uplift per sector (HVAC + AAFS + EV)."""
        return {
            "RES": self.d_hvac_res + self.d_aafs_res + self.d_ev_res,
            "COM": self.d_hvac_com + self.d_aafs_com + self.d_ev_com,
            "IND": self.d_hvac_ind + self.d_ev_ind,
            "DC":  self.d_dc,
            "LAG": self.d_residual,
        }


def compute_sector_uplift(
    yr: int,
    peak_row: "pd.Series",  # noqa: F821
    cal: SCMCalibration,
    sector_shares_yr: dict[str, float],
    scenario: str = "",
    dT_hub: float | None = None,
) -> SectorUpliftRow:
    """
    Compute disaggregated causal uplift for year `yr`.

    Parameters
    ----------
    yr               : projection year
    peak_row         : row from peak_tables[scenario].loc[yr]
    cal              : SCMCalibration (from calibrate())
    sector_shares_yr : {RES, COM, IND} fractional shares for year yr
    scenario         : scenario label for output tagging
    dT_hub           : override hub ΔT (default: uses value baked into cal)
    """
    from pipeline.config import OBSERVED_2022_UPLIFT_MW

    dT = dT_hub if dT_hub is not None else cal.dT_hub

    unadj  = float(peak_row["UNADJUSTED_CONSUMPTION"])
    aafs   = float(peak_row["AAFS"])
    lev    = float(peak_row["LIGHT_EV"])
    ldv    = float(peak_row["AATE_LDV"])
    dc     = float(peak_row["DATA_CENTER"])
    mnl    = float(peak_row["MANAGED_NET_LOAD"])

    # ── Sector UNADJ in MW ────────────────────────────────────────────────
    unadj_s = {s: unadj * sector_shares_yr[s] for s in ["RES", "COM", "IND"]}

    # ── HVAC structural uplift per sector ─────────────────────────────────
    # β_s × UNADJ_s(Y) × ΔT_hub, then scale by UNADJ_s(Y)/UNADJ_s(2025)
    # which simplifies to β_s × UNADJ_s(Y) × ΔT
    d_hvac = {s: cal.beta[s] * unadj_s[s] * dT for s in ["RES", "COM", "IND"]}

    # ── AAFS uplift (electrified building heat pumps) ─────────────────────
    d_aafs_res = aafs * AAFS_SECTOR_SPLIT["RES"] * cal.beta_aafs * dT
    d_aafs_com = aafs * AAFS_SECTOR_SPLIT["COM"] * cal.beta_aafs * dT

    # ── EV uplift ─────────────────────────────────────────────────────────
    ev_res = lev * LIGHT_EV_SECTOR_SPLIT["RES"]  + ldv * AATE_LDV_SECTOR_SPLIT["RES"]
    ev_com = lev * LIGHT_EV_SECTOR_SPLIT["COM"]  + ldv * AATE_LDV_SECTOR_SPLIT["COM"]
    ev_ind = lev * LIGHT_EV_SECTOR_SPLIT["IND"]  + ldv * AATE_LDV_SECTOR_SPLIT["IND"]
    d_ev_res = ev_res * cal.beta_ev * dT
    d_ev_com = ev_com * cal.beta_ev * dT
    d_ev_ind = ev_ind * cal.beta_ev * dT

    # ── DC thermal ────────────────────────────────────────────────────────
    d_dc = dc * cal.f_dc_thermal

    # ── Physical capacity ceiling (Validation 2) ──────────────────────────
    capacity_clipped = False
    clipped_sectors: list[str] = []
    clip_amount = 0.0

    d_hvac_clipped = {}
    for s in ["RES", "COM", "IND"]:
        ceiling = unadj_s[s] * cal.nameplate_cooling_fraction[s]
        raw     = d_hvac[s]
        if raw > ceiling:
            capacity_clipped = True
            clipped_sectors.append(s)
            clip_amount += raw - ceiling
            d_hvac_clipped[s] = ceiling
        else:
            d_hvac_clipped[s] = raw

    d_total = (
        sum(d_hvac_clipped.values())
        + d_aafs_res + d_aafs_com
        + d_ev_res + d_ev_com + d_ev_ind
        + d_dc
        + cal.d_residual
    )

    return SectorUpliftRow(
        year=yr,
        scenario=scenario,
        d_hvac_res=d_hvac_clipped["RES"],
        d_hvac_com=d_hvac_clipped["COM"],
        d_hvac_ind=d_hvac_clipped["IND"],
        d_aafs_res=d_aafs_res,
        d_aafs_com=d_aafs_com,
        d_ev_res=d_ev_res,
        d_ev_com=d_ev_com,
        d_ev_ind=d_ev_ind,
        d_dc=d_dc,
        d_residual=cal.d_residual,
        d_total=d_total,
        d_static=float(OBSERVED_2022_UPLIFT_MW),
        mnl_cec=mnl,
        mnl_hw=mnl + d_total,
        capacity_clipped=capacity_clipped,
        clipped_sectors=clipped_sectors,
        clip_amount_mw=clip_amount,
    )


