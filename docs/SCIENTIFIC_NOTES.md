# Scientific notes and known limits of v0.1 (global platform)

Read this before interpreting any output.

## 1. Wording decisions (enforced in code and reports)

* **Depicted facial expression**, not "emotion of the person". Sculptures are representations; Cowen & Keltner (2020) studied *perceived* expressions; Barrett et al. (2019) show even living faces do not map cleanly to emotions.
* **Implied facial-action configuration**, not "muscle movement detected". The AU table lists the musculature *associated* with each AU in living faces.
* **Morphological affinity to reference samples**, not "geographic origin". Millones' own PCA ellipses overlap across valleys; vessels are mould-made and may depict social types; manufacture place ≠ subject's origin.
* **UNKNOWN / ambiguous** is a first-class result. The classifier is allowed to decline.

## 2. FACS ≠ a 200+ emotion taxonomy

FACS (Ekman & Friesen 1978; 2002) describes ~44–46 Action Units. Combinations number in the thousands; EMFACS maps a small set of AU patterns to 6–7 basic categories. The MPI FACES database (Ebner et al. 2010) you linked is a *stimulus set* of 171 human faces × 6 expressions × 2 photos, not a 200-subcategory taxonomy. The platform therefore keeps three layers separate: geometry → AU-like configuration → expression category.

## 3. What is heuristic in v0.1 and how it gets calibrated

| Component | Status | Calibration path |
| --- | --- | --- |
| Landmarks | pretrained on human faces (MediaPipe) | validate landmark placement on ~50 sculptures; if poor, fine-tune a landmarker (e.g. on annotated crops) |
| Neutral template (14 ratios) | anthropometric approximations, single-image sanity check | `training/calibrate_neutral_template.py` on expert-labelled neutral vessels |
| AU evidence ramp (0.7–3.5 σ) | heuristic | fit ramp parameters to expert AU intensities |
| Expression prototypes | EMFACS literature | replace with `training/train_expression_classifier.py` model (group-split, ≥30 labelled samples, ideally ≥200) |
| Embedding | deterministic shape + HOG | replace with a learned embedding (e.g. ArcFace-style fine-tuned on sculptures) once ≥1k specimens exist |
| Quality thresholds | empirical | tune against annotator "usable / unusable" judgements |

## 4. Known failure modes

* **Headdresses, ear ornaments and face paint** can confuse the detector or the appearance embedding (Vergara 1992 catalogue). Use the face crop, and expect lower similarity reliability for painted faces.
* **Profile and three-quarter views**: AU estimates and frontal morphometrics degrade beyond ~30° yaw; the report warns and lowers pose quality. Millones' profile landmark set is a planned addition.
* **Damage / erosion** (missing noses, broken lips) shifts landmarks. Record damage in the validation comment; future: a damage mask.
* **Moulds**: near-identical vessels inflate any accuracy measured with image-level splits. Always use `training/split.py` and fill `mold_group`.
* **Style vs expression confound**: a workshop's habitual eye shape can be read as an AU. Stratify evaluation by museum/phase/valley (`training/evaluate.stratified_performance`) and inspect whether "expression" tracks style.
* **Small reference samples** (<30) make PCA axes and clusters unstable; the report says so.
* **Illumination sensitivity**: if the stability panel shows the top class changing across normalisers, treat the classification as unreliable regardless of confidence.

## 5. Quantities that are deliberately kept separate

`analysis_quality_score` (is the image analysable?) · `pose_quality` (is the view usable?) · `expression_confidence` (how peaked is the decision?) · `stability_score` (does the decision survive perturbation?) · `reliability` of affinity (margin between groups). Never collapse them into one number.

## 6. Relation to Millones' analysis

Same logic: landmarks → Procrustes → PCA → Ward clustering → grouping by realism / individuality / phase / valley. Differences: (a) landmarks are automatic (MediaPipe) rather than manually digitised in APS, so replicability improves but anatomical precision must be checked; (b) we add expression analysis, which Millones deliberately excluded by selecting neutral faces — the platform can reproduce his sample by filtering `expression == neutral`; (c) we quantify affinity with distances and margins instead of visual ellipse overlap, but report the same overlap caveat.

## 7. Ethics and data

Only public-domain / CC0 / CC BY(-SA) images are ingested automatically; each keeps licence and attribution. Museo Larco and other NC/ND collections require permission. The platform analyses objects, not living people, but the same wording discipline protects against biological-essentialist readings of "facial type" that appear in 1940s literature (Larco's "tipo Moche").

## 8. Additional rules for a multi-cultural corpus

* **Culture is never a feature.** The metadata firewall (`run_image_analysis`) makes it impossible to pass culture/region/period to the model; a test enforces it. Any future learned model must be trained only on the exported gold-standard feature vectors, which contain geometry and AU evidence but no taxonomy fields (they are carried alongside for *stratified evaluation* only).
* **Style vs expression.** Evaluate per `representation_type` and per `culture`; if accuracy tracks style categories more than annotator labels, the model is learning convention. The difficulty level and representation type fields exist for exactly this stratification.
* **External test by institution / culture.** `python training/train_expression_classifier.py --holdout "The Metropolitan Museum of Art"` or `--holdout-field culture --holdout moche` trains on everything else and tests on the held-out group.
* **Domain separation.** Photographs of living people, artworks, reconstructions, synthetic faces and human remains are separate `domain` values; the last three are excluded from training by default and the wording of the report changes for living persons.
* **Neutral/resting morphology vs expression.** The neutral template should eventually be estimated *per representation tradition* (a naturally narrow sculpted eye is not AU7); `calibrate_neutral_template.py` can be run on any filtered subset by exporting that subset's gold labels.
