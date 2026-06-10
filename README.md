# Instance-Aware Algorithm Configuration for E-CVRP

This repository contains the data, tuning artifacts, regression scripts, and
test results for:

> Yinghao Qin, Xinwei Wang, Mosab Bazargani, and Jun Chen. 2026.
> "Instance-Aware Parameter Configuration in Bilevel Late Acceptance Hill
> Climbing for the Electric Capacitated Vehicle Routing Problem."

The work combines Machine Learning (ML) and Operations Research (OR): it tunes
instance-specific parameters for Bilevel Late Acceptance Hill Climbing (b-LAHC),
then trains regression models that predict good b-LAHC parameters from EVRP
instance features before solving unseen instances.

- Paper: <https://arxiv.org/abs/2605.00572>
- CEC review details: [cec_pap158 Reviews Details.pdf](cec_pap158%20Reviews%20Details.pdf)

## Contents

| Path | Description |
| --- | --- |
| `data/` | 41 EVRP benchmark instances and raw feature data. |
| `feature-extraction/` | Notebook and cleaned feature table generation notes. |
| `instance-data-splitting/` | Fixed test split and 5-fold CV split generation. |
| `irace-two-stage-tuning/` | Rough and fine irace tuning artifacts for instance-specific labels. |
| `regression-training-prediction/` | CV, final model training, saved ridge models, and test prediction scripts. |
| `predicited-parameters-performance-on-test-set/` | Test-set parameter file and b-LAHC performance logs. |
| `evrp_instance_features_gpt.csv` | Final cleaned feature table used for regression. |
| `labels_median_rep.csv` | Instance-specific median and representative parameter labels. |
| `stats.xlsx` | Summary experiment statistics. |

## Workflow

1. Extract instance features from each `.evrp` file.
2. Split 41 instances into 33 training/CV instances and 8 held-out test
   instances.
3. Use two-stage irace tuning to obtain instance-specific labels for `his_len`
   and `max_attempts`.
4. Train regression models from features to parameter labels.
5. Predict parameters for held-out instances and evaluate them with b-LAHC.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn scipy joblib
```

## Reproduce Key Steps

Create the fixed test/CV split:

```bash
cd instance-data-splitting
python make_splits.py Instance.csv
```

Run regression cross-validation:

```bash
cd regression-training-prediction
python run_regression_cv.py \
  --features evrp_instance_features_gpt.csv \
  --labels labels_median_rep.csv \
  --splits splits.json \
  --target_mode median \
  --models ridge,rf \
  --out cv_results_median_fix.csv
```

Train final ridge models:

```bash
python train_final_models.py \
  --features evrp_instance_features_gpt.csv \
  --labels labels_median_rep.csv \
  --splits splits.json \
  --out_prefix final_ridge_fix \
  --target_mode median \
  --log_x \
  --log_y
```

Predict parameters for the held-out test instances:

```bash
python predict_test_params.py \
  --features evrp_instance_features_gpt.csv \
  --splits splits.json \
  --model_his final_ridge_fix_his_len.joblib \
  --model_att final_ridge_fix_max_attempts.joblib \
  --out predicted_params_test_fix.csv \
  --log_x
```

The existing held-out predictions are:

| Instance | `his_len` | `max_attempts` |
| --- | ---: | ---: |
| `E-n22-k4` | 1994 | 43 |
| `E-n112-k8-s11` | 1505 | 6 |
| `F-n140-k5-s5` | 4400 | 9 |
| `M-n212-k16-s12` | 2854 | 7 |
| `X-n221-k11-s7` | 6095 | 283 |
| `X-n469-k26-s10` | 5630 | 237 |
| `X-n698-k75-s13` | 5549 | 486 |
| `X-n1006-k43-s5` | 7018 | 503 |

## Notes

- The `target-runner` files in `irace-two-stage-tuning/` call an external
  b-LAHC executable from the original HPC environment. Update the `EXE` path in
  those files before rerunning irace elsewhere.
- The folder name `predicited-parameters-performance-on-test-set/` preserves the
  original experiment-artifact spelling.

## Citation

```bibtex
@article{qin2026instance,
  title={Instance-Aware Parameter Configuration in Bilevel Late Acceptance Hill Climbing for the Electric Capacitated Vehicle Routing Problem},
  author={Qin, Yinghao and Wang, Xinwei and Bazargani, Mosab and Chen, Jun},
  notes={has been accepted at IEEE CEC 2026},
  journal={arXiv preprint arXiv:2605.00572},
  year={2026}
}
```

## License

This repository is released under the MIT License. See `LICENSE` for details.
