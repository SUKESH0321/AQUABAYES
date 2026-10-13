#!/usr/bin/env python3
"""
AQUA / BAYES — Preprocessing Script
====================================
Orchestrates raw observation ingestion, leak-free chronological splitting,
standards-based WQI calculation, multi-target label generation, and quality reporting.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.data.pipeline import run_pipeline

if __name__ == '__main__':
    print("Starting AQUA / BAYES trustworthy data pipeline execution...")
    result = run_pipeline()
    qr = result['quality_report']
    print(f"Data pipeline complete. Total station observations: {qr['total_station_observations']} across {qr['total_daily_samples']} days.")
    print(f"Chronological split: Train={qr['splits']['train_station_observations']} | Val={qr['splits']['val_station_observations']} | Test={qr['splits']['test_station_observations']}")
    print(f"Station class distribution: {qr['class_distribution']['station_all']}")
    print(f"Daily class distribution: {qr['class_distribution']['daily_all']}")
    print(f"Thresholds fitted strictly on Train split without lookahead leakage.")
