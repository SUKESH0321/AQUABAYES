"""
AQUA / BAYES — Hydrological Standards and Water Quality Index (WQI) Specifications
===================================================================================
Incorporates definitions from:
- Bureau of Indian Standards (BIS) IS 10500: 2012 (Drinking Water Specification)
- World Health Organization (WHO) Guidelines for Drinking-water Quality (4th Edition)
- Central Pollution Control Board (CPCB) Designated Best Use Criteria (Classes A–E)
- Horton (1965) and Brown et al. (1970) Weighted Arithmetic Water Quality Index (WQI)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

# Jurisdictional and international reference standards summary
STANDARDS_METADATA = {
    'BIS_IS_10500_2012': {
        'title': 'Drinking Water — Specification (Second Revision)',
        'agency': 'Bureau of Indian Standards (BIS)',
        'parameters': {
            'ph': {'acceptable_min': 6.5, 'acceptable_max': 8.5, 'permissible_min': 6.5, 'permissible_max': 8.5, 'unit': 'pH scale'},
            'dissolved_oxygen': {'guideline_min': 5.0, 'unit': 'mg/L', 'note': 'Ecological and aesthetic indicator'},
            'specific_conductance': {'proxy_tds_mg_l': 500.0, 'permissible_tds_mg_l': 2000.0, 'unit': 'µS/cm'},
        }
    },
    'CPCB_DESIGNATED_BEST_USE': {
        'title': 'Water Quality Criteria for Designated Best Use',
        'agency': 'Central Pollution Control Board, India',
        'classes': {
            'A': 'Drinking Water Source without conventional treatment but after disinfection (pH 6.5–8.5, DO ≥ 6 mg/L, BOD ≤ 2 mg/L)',
            'B': 'Outdoor bathing organized (pH 6.5–8.5, DO ≥ 5 mg/L, BOD ≤ 3 mg/L)',
            'C': 'Drinking water source after conventional treatment and disinfection (pH 6.5–8.5, DO ≥ 4 mg/L, BOD ≤ 3 mg/L)',
            'D': 'Propagation of Wildlife and Fisheries (pH 6.5–8.5, DO ≥ 4 mg/L, Free Ammonia ≤ 1.2 mg/L)',
            'E': 'Irrigation, Industrial Cooling, Controlled Waste disposal (pH 6.0–8.5, Electrical Conductivity ≤ 2250 µS/cm)',
        }
    },
    'WHO_GDWQ_2022': {
        'title': 'Guidelines for Drinking-water Quality, 4th edition',
        'agency': 'World Health Organization (WHO)',
        'parameters': {
            'ph': {'optimal_min': 6.5, 'optimal_max': 8.5, 'unit': 'pH scale'},
            'dissolved_oxygen': {'note': 'Depletion indicates high microbial/organic pollution', 'unit': 'mg/L'},
        }
    }
}

# Supported feature definitions aligned with normalized source dataset (Zhao 2019)
# In Zhao 2019 archive, continuous parameters are normalized into [0, 1] relative to station dynamic ranges.
FEATURE_SPECS = {
    'ph': {
        'name': 'pH Index Midpoint',
        'description': 'Midpoint between minimum and maximum normalized daily pH indices across monitoring stations.',
        'optimal_range': (0.35, 0.65),  # Centered around neutral pH (7.0-7.5) in normalized scale
        'severe_deviation_bounds': (0.20, 0.85),
        'weight': 0.30,  # WQI parameter relative weight
    },
    'conductance': {
        'name': 'Specific Conductance Index',
        'description': 'Normalized electric conductivity representing dissolved mineral ions and salinity.',
        'optimal_range': (0.0, 0.40),  # Lower conductivity is cleaner freshwater
        'severe_deviation_bounds': (0.0, 0.75),
        'weight': 0.20,
    },
    'dissolved_oxygen': {
        'name': 'Dissolved Oxygen Index',
        'description': 'Normalized dissolved oxygen concentration indicating ecological vitality and aerobic status.',
        'optimal_range': (0.60, 1.0),  # High DO is vital for healthy aquatic ecosystem and low organic load
        'severe_deviation_bounds': (0.30, 1.0),  # Below 0.30 represents hypoxia / high BOD
        'weight': 0.35,
    },
    'temperature': {
        'name': 'Thermal Index',
        'description': 'Normalized water temperature influencing oxygen solubility and microbiological growth kinetics.',
        'optimal_range': (0.20, 0.60),  # Moderate temperate range
        'severe_deviation_bounds': (0.0, 0.85),
        'weight': 0.15,
    }
}

RISK_CLASSES = ['low', 'moderate', 'high', 'critical']


def compute_sub_index(feature: str, value: float) -> float:
    """
    Computes quality rating sub-index q_i (0 to 100) for a normalized parameter value.
    Higher q_i indicates higher water quality / compliance.
    """
    val = float(np.clip(value, 0.0, 1.0))
    spec = FEATURE_SPECS[feature]
    opt_min, opt_max = spec['optimal_range']
    sev_min, sev_max = spec['severe_deviation_bounds']

    if feature == 'ph':
        # pH is optimal in the middle; penalize deviation both low (acidic) and high (alkaline)
        if opt_min <= val <= opt_max:
            return 100.0
        elif val < opt_min:
            if val <= sev_min:
                return max(0.0, (val / max(1e-4, sev_min)) * 35.0)
            else:
                return 35.0 + ((val - sev_min) / max(1e-4, opt_min - sev_min)) * 65.0
        else:  # val > opt_max
            if val >= sev_max:
                return max(0.0, ((1.0 - val) / max(1e-4, 1.0 - sev_max)) * 35.0)
            else:
                return 35.0 + ((sev_max - val) / max(1e-4, sev_max - opt_max)) * 65.0

    elif feature == 'conductance':
        # Conductance is best when low (freshwater)
        if val <= opt_max:
            return 100.0 - (val / opt_max) * 20.0  # 80-100
        elif val <= sev_max:
            return 80.0 - ((val - opt_max) / (sev_max - opt_max)) * 50.0  # 30-80
        else:
            return max(0.0, 30.0 - ((val - sev_max) / (1.0 - sev_max)) * 30.0)  # 0-30

    elif feature == 'dissolved_oxygen':
        # Dissolved oxygen is best when high; low DO is severe hypoxia
        if val >= opt_min:
            return 80.0 + ((val - opt_min) / (1.0 - opt_min)) * 20.0  # 80-100
        elif val >= sev_min:
            return 30.0 + ((val - sev_min) / (opt_min - sev_min)) * 50.0  # 30-80
        else:
            # Acute hypoxia
            return max(0.0, (val / max(1e-4, sev_min)) * 30.0)  # 0-30

    elif feature == 'temperature':
        # Thermal stress
        if opt_min <= val <= opt_max:
            return 100.0
        elif val < opt_min:
            return 70.0 + (val / opt_min) * 30.0
        else:
            return max(20.0, 100.0 - ((val - opt_max) / (1.0 - opt_max)) * 70.0)

    return 70.0


def calculate_wqi(record: Dict[str, float]) -> float:
    """
    Computes Horton/Brown Weighted Arithmetic Water Quality Index (WQI).
    WQI in [0, 100], where 100 is pristine water quality and 0 is extreme contamination.
    """
    total_weight = sum(spec['weight'] for spec in FEATURE_SPECS.values())
    weighted_sum = sum(
        spec['weight'] * compute_sub_index(feat, record.get(feat, 0.5))
        for feat, spec in FEATURE_SPECS.items()
    )
    return float(np.clip(weighted_sum / total_weight, 0.0, 100.0))


def classify_risk_state(wqi: float, record: Dict[str, float] = None) -> str:
    """
    Assigns rigorous multi-state risk class:
    - 'low': WQI >= 75 (Good/Excellent water condition)
    - 'moderate': 55 <= WQI < 75 (Fair water condition, investigation/monitoring advised)
    - 'high': 35 <= WQI < 55 (Poor water condition, treatment necessary)
    - 'critical': WQI < 35 or acute single-parameter failure (e.g. acute hypoxia DO < 0.20 or pH < 0.15)
    """
    # Acute single-parameter safety check (precautionary principle)
    if record is not None:
        do_val = record.get('dissolved_oxygen', 1.0)
        ph_val = record.get('ph', 0.5)
        cond_val = record.get('conductance', 0.0)
        if do_val < 0.18 or ph_val < 0.12 or ph_val > 0.92 or cond_val > 0.85:
            return 'critical'

    if wqi >= 75.0:
        return 'low'
    elif wqi >= 55.0:
        return 'moderate'
    elif wqi >= 35.0:
        return 'high'
    else:
        return 'critical'


def determine_hazards(record: Dict[str, float]) -> Dict[str, Any]:
    """
    Determines specific environmental and chemical hazard flags.
    """
    ph_val = float(record.get('ph', 0.5))
    cond_val = float(record.get('conductance', 0.0))
    do_val = float(record.get('dissolved_oxygen', 0.8))

    chem_hazard = int(ph_val < 0.25 or ph_val > 0.75 or cond_val > 0.50)
    bio_hazard = int(do_val < 0.40)  # Low DO signifies elevated organic decomposition / pathogen load
    acute_exceedance = int(chem_hazard or bio_hazard)

    return {
        'chemical_hazard': chem_hazard,
        'biological_hazard': bio_hazard,
        'acute_exceedance': acute_exceedance
    }
