# Datasets

## 0. Global source corpora (connectors in `scripts/connectors/museums.py`)

| Source | Coverage | API key | Rights handling | Connector |
| --- | --- | --- | --- | --- |
| Europeana | thousands of European institutions, prehistoric → contemporary, 8k+ 3-D items | free key | `rights` field normalised; only open reusability queried | `europeana` |
| Smithsonian Open Access | global, 2-D + 3-D, CC0 filter | api.data.gov key | CC0 only | `smithsonian` |
| J. Paul Getty Museum | antiquities → modern, Linked Art + IIIF | none | Linked Art `subject_to` rights | `getty` |
| The Met | global, Open Access | none | `isPublicDomain` | `met` |
| Rijksmuseum | 800k records, 600k hi-res, IIIF | none (search) | CC0 | `rijksmuseum` |
| Art Institute of Chicago | global, IIIF, data dumps | none | `is_public_domain` | `artic` |
| Harvard Art Museums | global | free key | `imagepermissionlevel`/`copyright` | `harvard` |
| Wikimedia Commons | high-recall global photographs | none | per-file licence; NC/ND skipped | `wikimedia` |
| British Museum, Brooklyn, Princeton, Museo Larco, national museums | manual / agreement | — | `import_local.py` with per-row licence | `LOCAL` |

Image access via IIIF (Image API 3.0) wherever the institution supports it; access ≠ training rights, so the policy in `config/taxonomy.json → license` decides `training_allowed` per object.


## 1. Moche-specific sources (the Moche collection inside this platform)

| Source | Licence | Access | Ingestion | Notes |
| --- | --- | --- | --- | --- |
| The Metropolitan Museum of Art — Open Access | CC0 for objects flagged `isPublicDomain` | public API, no key | `scripts/ingest_met.py` | Moche portrait heads / stirrup-spout bottles (3rd–8th c.); rich fields: culture, period, date range, geography, medium, credit line |
| Art Institute of Chicago | CC0 for `is_public_domain` | public API + IIIF | `scripts/ingest_artic.py` | ~100 Moche objects; `place_of_origin` often carries the valley |
| Wikimedia Commons | per file (CC0 / PD / CC BY / CC BY-SA kept; NC/ND skipped) | public API | `scripts/ingest_wikimedia.py` | supplementary photographs (Fowler, Larco, Quai Branly…); museum parsed from description |
| Museo Larco online catalogue | CC BY-NC-ND 4.0 (research use with attribution) | request permission for ML use | `scripts/import_local.py` | the single most important collection (655 portrait vessels in Millones' count) with valley + Larco phase metadata |
| Princeton University Art Museum | public-domain downloads for many objects | manual download | `scripts/import_local.py` | portrait vessels incl. Gillett G. Griffin bequest |
| Brooklyn Museum | CC BY (many objects) | API (key) / manual | `scripts/import_local.py` | Moche objects |
| Smithsonian Open Access | CC0 | API / bulk | manual / future script | includes 3-D scans (OBJ/GLB) → phase-2 3-D morphometrics |
| Museo Nacional de Arqueología, Antropología e Historia del Perú; Museo de Arte de Lima (MALI); Museo Cassinelli; Museum für Völkerkunde Berlin; Linden-Museum Stuttgart; British Museum (CC BY-NC-SA) | varies | request | `import_local.py` | large Moche holdings worth a data-sharing request |

`import_local.py` CSV columns: `filename,id,museum,object_number,title,culture,period,date,valley,phase,provenance,material,source_url,license,photographer,expression,expression_source,mold_group`.

## 2. Human facial datasets (for pre-training / benchmarking, never as the final model's domain)

| Dataset | Content | Why useful | Access |
| --- | --- | --- | --- |
| CMU Multi-PIE | 337 subjects, 15 views, 19 illuminations, 6 expressions | pose + illumination robustness tests | licence agreement |
| Radboud Faces (RaFD) | 67 models, 8 expressions, 5 camera angles, FACS-trained posers | expression × angle | research request |
| Bosphorus | 3-D, 105 subjects, AU-coded, poses, occlusions | 3-D AU ground truth | research request |
| D3DFACS | dynamic 3-D, FACS-coded sequences | 3-D AU dynamics | research request |
| RIKEN multi-angle | dynamic, multiple angles, 29 manually coded AUs | AU × angle | research request |
| CK+, DISFA, BP4D | AU intensity labels (2-D) | training an AU regressor | research request |
| AffectNet | ~1 M in-the-wild faces, 8 categories + valence/arousal | large categorical pre-training | research request |
| FER-2013 | 35k 48×48 faces, 7 categories | quick baselines only (noisy) | Kaggle |

**Rule:** Human data → pre-train / sanity-check → fine-tune on expert-labelled sculpture. Tian et al. (2021) showed direct transfer fails on sculpture.

## 3. GlobalFaceCorpus structure and versioning

```
data/mochefacedb/
  images/<SOURCE>_<ID>/original.<ext>   untouched download
  images/<SOURCE>_<ID>/thumb.jpg
  metadata.jsonl                        one record per object (all fields above + expression labels)
  licenses.jsonl                        licence + attribution + URLs per image
  VERSION                               "GlobalFaceCorpus v0.N" (bumped by every ingestion)
  CHANGELOG.md
data/reference_index/
  index.json  embeddings.npy  landmarks.npy  skipped.json
```

Every analysis report records `versions.dataset` = the `VERSION` string in force when the index was built.

## 4. Annotation protocol (recommended before v0.2)

Per specimen, two independent FACS-trained annotators record: depicted expression (7 + ambiguous), visible AU-like actions (present / intensity A–E), damage / repainting, stylisation level (Millones/de la Fuente: esquema · convención · retrato · hiper-retrato), pose usability, image quality, `mold_group` if suspected. Report inter-rater agreement (Cohen's κ) with the dataset version. Store as `expression`, `expression_source="expert"`, and `corrected_aus` fields.
