"""Chapter 3.2 PSM-DID (Equation 3).

Y_ict^{PSM} = β0 + β1 (treat_i × time_t) + β2 X_ict + μ_c + θ_t + ε_ict

The outcome network has the same structure as TimeVaryingDID; it is estimated
on the matched sample obtained from the Logit propensity scores.
"""

from __future__ import annotations

from src.models.time_varying_did import TimeVaryingDID


class PSMDID(TimeVaryingDID):
    """DID network restricted to the propensity-score-matched sample (Eq. 3)."""

    pass
