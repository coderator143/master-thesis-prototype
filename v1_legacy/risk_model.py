"""
Accident-risk KPI model.

Implements the causal graph and weighted scoring model described in the
thesis (Problem Statement / Proposed Methodology chapters): four scene
variables (speed, visibility, proximity, weather) feed into a single
weighted "accident risk" score, and interventions can be simulated by
overriding one or more variables and recomputing the score.

No file I/O and no printing here on purpose -- this module is the "causal
model", kept separate from prototype.py (the "simulation harness") so it
can be read, tested, and explained on its own.
"""

# Category -> score mapping. Follows the low/medium/high pattern used in
# the thesis's own prototype code sketch (Fig 6.1), extended to weather.
SPEED_SCORE = {"low": 0.2, "medium": 0.5, "high": 0.9}
VISIBILITY_SCORE = {"high": 0.2, "medium": 0.5, "low": 0.9}
PROXIMITY_SCORE = {"far": 0.2, "medium": 0.5, "close": 0.9}
WEATHER_SCORE = {"clear": 0.2, "rainy": 0.5, "night": 0.9}

# Contribution weights for each variable (must sum to 1.0).
# Reconciled from the thesis's DAG figure (Fig 3.1), whose raw weights
# (weather 0.41, visibility 0.29, speed 0.21, proximity 0.29) summed to
# 1.20 instead of 1.0 -- normalized here by dividing by 1.20 and rounding,
# preserving the original ordering (weather highest, speed lowest).
DEFAULT_WEIGHTS = {
    "weather": 0.35,
    "visibility": 0.25,
    "proximity": 0.25,
    "speed": 0.15,
}

# Risk score band boundaries, per the thesis's Table 5.1.
RISK_BANDS = [
    (0.3, "Low Risk"),
    (0.6, "Moderate Risk"),
    (1.01, "High Risk"),
]


def calculate_risk(speed, visibility, proximity, weather, weights=DEFAULT_WEIGHTS):
    """Weighted accident-risk score in [0, 1], per the thesis's KPI formula:
    Risk Score = w1(speed) + w2(visibility) + w3(proximity) + w4(weather).
    """
    score = (
        weights["speed"] * SPEED_SCORE[speed]
        + weights["visibility"] * VISIBILITY_SCORE[visibility]
        + weights["proximity"] * PROXIMITY_SCORE[proximity]
        + weights["weather"] * WEATHER_SCORE[weather]
    )
    return round(score, 2)


def interpret_risk(score):
    """Maps a risk score to a Low/Moderate/High Risk label."""
    for threshold, label in RISK_BANDS:
        if score < threshold:
            return label
    return RISK_BANDS[-1][1]


def simulate_intervention(base_scenario, changes, weights=DEFAULT_WEIGHTS):
    """Simulates an intervention on a scenario.

    base_scenario: dict with keys speed/visibility/proximity/weather.
    changes: dict of overrides to apply on top of base_scenario, e.g.
        {"speed": "low"} to simulate "what if speed had been lower".

    Returns a dict with the original and intervened scenarios, their
    risk scores and levels, and the score delta (new - base).
    """
    new_scenario = {**base_scenario, **changes}
    base_score = calculate_risk(**base_scenario, weights=weights)
    new_score = calculate_risk(**new_scenario, weights=weights)
    return {
        "base_scenario": base_scenario,
        "new_scenario": new_scenario,
        "base_score": base_score,
        "new_score": new_score,
        "base_level": interpret_risk(base_score),
        "new_level": interpret_risk(new_score),
        "delta": round(new_score - base_score, 2),
    }
