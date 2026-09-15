# SIB Cathode — ML-Assisted Screening for Prussian White Cathode Materials

An independent materials-informatics project exploring whether ML can predict sodium-ion battery cathode performance before committing candidates to full experimental iteration.

Inspired by [Altris AB](https://www.altris.se)'s Prussian White technology. Not affiliated with Altris, Clarios, or Volvo Cars.

---

## The Problem

Experimental iteration in battery materials development is slow and expensive. A new cathode composition can require synthesis, electrode fabrication, cell assembly, cycling, and characterization before you know if it's worth pursuing.

This project asks: can a model trained on known compositions give R&D teams a useful signal before that work begins?

---

## What's in This Repo

| File | Description |
|------|-------------|
| `sib_cathode_dummy_dataset.csv` | 50-row synthetic dataset mirroring real Prussian White experimental data |
| `sib_cathode_xgboost_baseline.ipynb` | XGBoost multi-target regression baseline |
| `sib_cathode_piml_hybrid.ipynb` | Physics-informed hybrid model (Faraday + NN) |
| `artifacts/sib_xgb_baseline_v1.joblib` | Trained XGBoost pipeline |
| `artifacts/sib_xgb_baseline_v1_metadata.json` | Training metadata and per-target metrics |

---

## Dataset

Synthetic dataset structured to mirror real Prussian Blue Analog (PBA) experimental data. 50 samples across 11 chemistry families:

- Pure PW, Mn-sub, Cu-sub, Co-sub, Zn-sub, Multi-metal
- Vacancy variants, Hydrated, Elevated-T, Alt Electrolyte, V-Window

**82 columns** spanning composition, synthesis conditions, electrochemical protocol, Materials Project DFT features, Matminer descriptors, and domain PBA priors.

**5 prediction targets:** specific capacity (mAh/g), capacity retention (%), average voltage (V), initial Coulombic efficiency (%), rate capability (1C/5C %).

> These metrics validate the pipeline, not the chemistry. The interesting question is whether the modeling approach holds up when real experimental data replaces the synthetic set.

---

## Models

### XGBoost Baseline

68 engineered features. Composition-grouped train/test split to prevent leakage between chemically similar formulas. Hyperparameter tuning with grouped cross-validation.

**Test set results (specific capacity):**

| Metric | Value |
|--------|-------|
| MAPE | 4.4% |
| R² | 0.90 |
| MAE | 5.8 mAh/g |

**Per-target reality check:**

| Target | MAPE | R² | Verdict |
|--------|------|----|---------|
| Specific capacity | 4.4% | 0.90 | Works |
| Capacity retention | 6.0% | 0.53 | Marginal |
| Avg voltage | 2.8% | -0.06 | Fails |
| ICE | 1.4% | -1.03 | Fails |
| Rate capability | 2.7% | -0.07 | Fails |

Low MAPE ≠ good model. Always check R².

### Physics-Informed Hybrid (PIML)

Starts with Faraday's Law, not with data.

```
Q_predicted = Q_theoretical × u_vacancy × u_water × u_Na × u_kinetic(NN) + residual(NN)
```

- `Q_theoretical` — computed exactly from composition (no ML)
- `u_deterministic` — vacancy, water, Na penalties from chemistry rules
- `u_kinetic` — learned by neural network, constrained to [0.55, 1.05]
- Hard constraint: no prediction exceeds theoretical maximum

The model only learns what physics can't explain. Each prediction comes with a decomposed utilization factor — a chemically interpretable reason for why a composition underperforms its theoretical ceiling.

---

## Key Finding

The model succeeds where physics is well-understood (capacity) and fails where it isn't (voltage, ICE). That's not a limitation — it's a map of where experimental effort still needs to go.

Top predictors, in order:
1. Na stoichiometry — directly bounds extractable charge
2. Vacancy fraction — missing framework sites = lost capacity
3. C-rate — kinetics dominate at fast discharge
4. Theoretical capacity (Faraday-derived)
5. Jahn-Teller activity, ionic radius mismatch

The model didn't need to be told any of this. It found it in the data.

---

## What's Next

- Extrapolation experiment: train on Fe/Mn systems, predict Cu/Ni chemistry the model has never seen
- SHAP analysis by chemistry family
- Honest assessment of where this is and isn't trustworthy for experimental screening

---

## Stack

```
Python 3.10+
xgboost, scikit-learn, pytorch
pandas, numpy, matplotlib, seaborn
matminer (feature generation)
```

---

## Author

**Hussnain Tariq** — Independent engineering project.
[LinkedIn](https://www.linkedin.com/in/hussnain-tariq-02454934a/)

---

*This is not an Altris, Clarios, or Volvo Cars project. Built from publicly available information as an independent exploration of ML-assisted materials screening.*
