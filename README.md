# cxr-evidence-pack

An independent, reproducible **technical-performance evaluation** of an open chest X-ray classifier, written the
way a notified body would read it: structured on the Team-NB *Questionnaire: Artificial Intelligence in Medical
Devices* (v1, Nov 2024) and pillar 2 of MDCG 2020-1 (technical performance).

> Research artefact. Not a medical device, not clinical advice, no claim of regulatory conformity.

## What is evaluated

| | |
|---|---|
| Model | TorchXRayVision `densenet121-res224-chex` (DenseNet-121, trained on CheXpert only) |
| Test images | NIH ChestX-ray14 (NIH Clinical Center, CC0): an external site for this model |
| Reference standard | Google Health radiologist-adjudicated labels: 4 findings (1.962 test images), all findings + normal/abnormal (810 PA test images) |

## Contents (filled milestone by milestone)

- `docs/data-dossier.md`: provenance, licences, labelling process, exclusions, label mapping
- `docs/evaluation-report.md`: metrics with 95% CIs, subgroups, calibration, best/worst cases, operating boundaries, baseline, explainability checks
- `docs/model-card.md`: intended use, limitations, change-control plan
- `docs/monitoring-plan.md`: post-market performance monitoring template
- `docs/traceability.md`: each Team-NB question → where it is answered

## Reproduce

Runs on a free Kaggle notebook (CPU is enough) with the NIH dataset attached. The labels need Google's access form. See `scripts/01_inference.py`.

## Data attribution

NIH ChestX-ray14: Wang X. et al., *ChestX-ray8*, CVPR 2017; data provided by the NIH Clinical Center
(https://nihcc.app.box.com/v/ChestXray-NIHCC). Expert labels: Majkowska A. et al., *Radiology* 2020; Nabulsi Z. et al.,
*Scientific Reports* 2021. Model: Cohen J.P. et al., TorchXRayVision. No images or labels are redistributed here.

## Author

Sami LABBANE DIT KALTI. Code under Apache-2.0.
