# Data dossier

Status: **draft**. The four-findings label file has been received and checked (7 Oct 2026); counts in §7–8 are
filled from `results/run.json` at M1.

This dossier answers the data-management questions of the Team-NB *Questionnaire: Artificial Intelligence in
Medical Devices* (v1, Nov 2024), §6.3, for an evaluation-only study: no model is trained or fine-tuned here, so
the data serve as **test data** (and, for calibration and threshold choice, as **tuning data**), never as
training data.

## 1. Data used, at a glance

| Role | Data | Images | Source |
|---|---|---|---|
| Test, task 1 | NIH ChestX-ray14 images × Google four-findings expert labels, test split | 1.962 | NIH Clinical Center; Google Health |
| Tuning (temperature, thresholds) | Same, validation split | 2.414 | idem |
| ~~Test, task 2~~ | NIH ChestX-ray14 PA images × Google all-findings expert labels | 810 | **Not available** (§2): out of scope for this version |
| Model under test | TorchXRayVision `densenet121-res224-chex` | — | Cohen et al.; trained on CheXpert (Stanford) |

No image or label is redistributed in this repository. Anyone can rebuild the analysis from the original
sources (§2) with `scripts/01_inference.py` and `scripts/02_evaluate.py`.

## 2. Provenance

| Item | Origin | How obtained | Version / integrity |
|---|---|---|---|
| Images | NIH Clinical Center, Bethesda (USA). ChestX-ray14: 112.120 frontal chest radiographs of 30.805 patients, PNG 1024×1024, 8-bit, converted from the hospital PACS (Wang et al. 2017) | Kaggle dataset `nih-chest-xrays/data`, version 3 (mirror of the NIH Box release): only the 4.376 labelled images are downloaded, file by file (`scripts/00_download_images.py`) | Size and SHA-256 per image in `data/nih/manifest.csv`; download date recorded at M1 |
| Four-findings labels | Google Health; Majkowska et al. 2020 | Copy bundled in the TorchXRayVision repository: `torchxrayvision/data/google2019_nih-chest-xray-labels.csv.gz` (last changed in commit `8ef8846141`, 22 Jun 2021). Google's own Cloud Storage bucket serves an approval list only: on 7 Oct 2026 it refused read access to an authenticated account with a billing project, and the access form linked from the dataset page returned "file not found" | SHA-256 `1d1b846e463753ed0219f67e21b10c8aef381897dbd64865c8191aa5794e8048`, also recorded in `run.json` |
| All-findings labels | Google Health; Nabulsi et al. 2021 | Not obtained: only in Google's restricted bucket | — |
| Image metadata (sex, age, view, follow-up number) | NIH `Data_Entry_2017.csv`, or the metadata columns carried in Google's label files | With the images | idem |
| Model weights | Cohen et al., TorchXRayVision; DenseNet-121 trained on CheXpert | `pip install torchxrayvision`; weights downloaded by the library | Library version pinned at M1; as-shipped thresholds copied from source commit `6e9c706ed1` (`configs/shipped_thresholds.yaml`) |

## 3. Licence and permission chain

| Item | Terms | Consequence here |
|---|---|---|
| NIH images | Google Cloud documentation: "no restrictions on the use of the NIH chest x-ray images"; attribution required: link to the NIH download site and cite Wang et al., CVPR 2017. Commonly listed as CC0 | Used; attributed in README and report; not redistributed |
| Google expert labels | The dataset page **states no licence for the labels**; it requires citing Majkowska 2020 (and Nabulsi 2021 for the all-findings set). Google's request form asks only for affiliation type and purpose, with no terms to accept. The copy used here is redistributed by TorchXRayVision (Apache-2.0 repository) | Used for evaluation only; cited; not redistributed here; only aggregate results are published |
| TorchXRayVision code | Apache-2.0 | Used |
| `chex` weights | Trained on CheXpert, distributed under the Stanford Research Use Agreement (non-commercial research) | The pack is a free research artefact: nothing is sold, the weights are not redistributed. A client engagement applies the *method* to the client's own model and data, never these weights |
| CheXpert, MIMIC-CXR, PadChest, VinDr-CXR | Research-only agreements | Not used |

## 4. Reference standard and labelling process

**Four findings** (Majkowska et al. 2020). Findings: airspace opacity, pneumothorax, nodule or mass, fracture.
Each image was read by **three radiologists**; when all agreed after the first read, that label was final;
otherwise the readers adjudicated until consensus, for **up to five rounds**. Readers could answer `YES`, `NO`, or `HEDGE` (uncertain), the latter **only for nodule/mass and pneumothorax**;
adjudication resolves every image to **`YES` or `NO`**, and the file used here contains only adjudicated values
(no missing or HEDGE entries). Individual reads exist only in Google's restricted bucket, so reader agreement
is not reported in this version.

The file holds both splits (`Set Id`): **1.962 test** images, matching Google's count, and **2.414 validation**
images, two more than the 2.412 stated on Google's dataset page. The two-image difference is reported, not
corrected; no subset of either split is used for training.

**All findings** (Nabulsi et al. 2021) — *described for completeness; not used in this version.* 810 PA images from the four-findings test split. **Five board-certified
radiologists** (American Board of Radiology) read each image independently. Three of the five were chosen at
random as "ground-truth radiologists"; the final label of each finding and of normal/abnormal is their
**majority vote**. Values: `YES`, `NO`. An `Other` column is `YES` when the majority saw a finding outside the
14, or called the image abnormal without any single finding reaching a majority.

The reference standard is therefore **image-only expert reading**: no reports, CT, follow-up or pathology. It
measures agreement with adjudicated radiologists, not with clinical truth.

## 5. Independence of the test data from the model

- **Different site:** the model was trained on CheXpert (Stanford Hospital); the test images come from the NIH
  Clinical Center. No NIH image was used to train, select or tune the model.
- **Different labelling method:** CheXpert training labels were extracted by an NLP labeller from radiology
  reports; the test labels are radiologist reads of the image. Part of any performance gap is a gap in **label
  definition**, not in the model (§6).
- **No tuning on test data:** temperature and the 90%-sensitivity thresholds are fitted on the 2.414-image
  validation split and applied unchanged to the test split. The as-shipped thresholds come from the library.
- **No patient overlap** between the splits: 860 test patients and 835 validation patients, **0 in both**
  (checked on the label file, 7 Oct 2026; re-checked in `run.json`, `patients_in_both_validation_and_test`).

## 6. Label mapping (reference standard → model output)

Configured in `configs/label_map.yaml`. Model outputs checked against the TorchXRayVision source
(`model_urls['chex']`).

| Reference label | Model output | Fit of definitions |
|---|---|---|
| Airspace opacity | Lung Opacity | CheXpert "Lung Opacity" is broader (it includes e.g. atelectasis-type opacities): expect lower specificity |
| Pneumothorax | Pneumothorax | Direct |
| Nodule or mass | Lung Lesion | CheXpert "Lung Lesion" covers nodule, mass and other focal lesions: close, not identical |
| Fracture | Fracture | CheXpert fracture is any fracture mentioned in the report; Google's covers fractures visible on the image |
| Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Pneumonia, Pneumothorax (all-findings set, not used in this version) | Same names | Direct names; "Pneumonia" is partly a clinical, not radiological, label in both sources |
| Normal/abnormal | Maximum logit over all model outputs | The model has no "No finding" output: a derived score, reported as exploratory |

Not mapped in the all-findings set (for a later version): emphysema, fibrosis, hernia, infiltration, mass, nodule, pleural
thickening (no matching `chex` output, or covered only through "Lung Lesion", which task 1 already tests).

Column names confirmed on the label file: `Airspace opacity`, `Pneumothorax`, `Nodule or mass`, `Fracture`, plus
the NIH metadata (`Patient ID`, `Patient Age`, `Patient Gender`, `View Position`, `Follow-up #`) and `Set Id`.

## 7. Exclusions and missing data

| Rule | Applies to | Primary analysis | Sensitivity analysis |
|---|---|---|---|
| `HEDGE` label | — | Not applicable: the adjudicated file has no HEDGE values (§4) | — |
| Labelled image not found or unreadable | Any | Excluded; listed by name in `run.json` | — |
| Implausible age (> 110; one image, recorded as 155) | Age subgroup only | Age treated as missing; image kept | — |
| Missing metadata (age, sex, view) | Subgroup tables only | Image kept in the overall analysis, left out of that subgroup | — |
| Subgroup with < 5 positives or < 5 negatives | Subgroup AUROC | Not estimated; counts still reported | — |

No image-quality filter, no outlier removal, no manual review of images before evaluation. Counts after M1
(_to fill_):

| | Test | Validation |
|---|---|---|
| Labelled images | 1.962 | 2.414 |
| Patients | 860 | 835 |
| Not found / unreadable | | |
| Evaluated | | |

## 8. Population description (to fill at M1)

Per split: patients, images per patient, sex, age bands (<40, 40–59, 60–79, 80+), view (PA/AP), follow-up
number (0, 1–4, 5+), and prevalence of each finding. These are the strata of the subgroup analysis.

## 9. Known limitations of the data

- **One institution, one country, historical data:** NIH Clinical Center, USA; images acquired 1992–2015 (Wang
  et al. 2017). Not representative of European populations, current equipment or other care settings.
- **Reduced image fidelity:** 8-bit PNG at 1024×1024 instead of the original DICOM; the model then resizes to
  224×224.
- **Selection:** the Google splits were sampled by Google for its own studies; prevalences do not match a
  clinical population, so PPV and NPV do not transfer.
- **Repeated patients:** several images per patient; confidence intervals use a patient-level bootstrap.
- **No race or ethnicity, scanner or acquisition-site metadata** in NIH ChestX-ray14: fairness can be checked
  only by sex, age and view.
- **Reference standard:** image-only reads, see §4; definitional mismatch with the training labels, see §6.
- **Second-hand copy of the labels:** obtained from the TorchXRayVision repository, not from Google directly
  (§2). Counts match Google's documentation for the test split; the validation split differs by two images.
- **Four findings only:** the all-findings and normal/abnormal labels were not obtainable, so the per-finding
  results for atelectasis, cardiomegaly, effusion and the rest are not part of this version.

## 10. References

- Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. ChestX-ray8: hospital-scale chest X-ray database and
  benchmarks. CVPR 2017. Data: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Majkowska A, et al. Chest radiograph interpretation with deep learning models: assessment with
  radiologist-adjudicated reference standards and population-adjusted evaluation. *Radiology* 2020.
- Nabulsi Z, et al. Deep learning for distinguishing normal versus abnormal chest radiographs and generalization
  to two unseen diseases: tuberculosis and COVID-19. *Scientific Reports* 2021.
- Cohen JP, et al. TorchXRayVision: a library of chest X-ray datasets and models. MIDL 2022.
  https://github.com/mlmed/torchxrayvision
- Irvin J, et al. CheXpert: a large chest radiograph dataset with uncertainty labels and expert comparison.
  AAAI 2019.
- Google Cloud, NIH Chest X-ray dataset and expert labels:
  https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest (accessed 7 Oct 2026)
- Team-NB, Questionnaire: Artificial Intelligence in Medical Devices, v1, November 2024.
