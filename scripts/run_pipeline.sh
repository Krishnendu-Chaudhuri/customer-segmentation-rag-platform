#!/usr/bin/env bash
# Run the full data pipeline: etl -> features -> segmentation -> personalization
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
elif [[ -f ".venv/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/Scripts/activate"
fi

pip install -e . -q

echo "==> Module 1: ETL"
python -m shopper_segmentation.etl

echo "==> Module 2: Features"
python -m shopper_segmentation.features

echo "==> Module 3: Segmentation"
python -m shopper_segmentation.segmentation

echo "==> Module 4: Personalization"
python -m shopper_segmentation.personalization

echo "Pipeline complete."
