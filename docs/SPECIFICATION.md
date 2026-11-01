# Technical specification — Global Face Heritage Platform v0.1

See README.md for the design rules and docs/DATA_MODEL.md for the schema. This file lists the API.

| Method | Path | Role | Purpose |
| --- | --- | --- | --- |
| POST | `/api/auth/register`, `/api/auth/login`; GET `/api/auth/me` | — | accounts (first = admin) |
| GET | `/api/taxonomy`, `/api/taxonomy/stats` | any | controlled vocabularies, counts |
| POST/GET | `/api/objects` | any | create (validated) / list with filters `q, culture, region, domain, period, license, mine` |
| GET/PATCH/DELETE | `/api/objects/{id}` | any / owner-admin | object with images, faces, region path, collections |
| POST | `/api/objects/{id}/images` | any | upload + analyse; returns report, faces, duplicate hints |
| POST | `/api/analyse` | any | one-shot: create object + upload + analyse |
| GET | `/api/images`, `/api/images/{id}`, `/{id}/artefact/{path}`, `/{id}/download?format=json|csv|pdf`; POST `/{id}/rerun`; DELETE | any | results, artefacts, reports, re-run |
| POST/GET | `/api/faces/{id}/annotations` | any (expert layer: FACS-certified) | human annotations |
| GET | `/api/references`, `/stats`, `/{id}/image` | any | reference corpus |
| GET/PATCH | `/api/admin/users[/{id}]` | admin | roles, activation, FACS certification |
| GET | `/api/admin/stats`, `/flags`, PATCH `/flags/{id}`, GET `/audit` | admin | monitoring, review queue |
| GET/POST | `/api/admin/collections`, POST `/collections/{id}/refresh` | admin | versioned research subsets |
| POST | `/api/admin/duplicates/scan` | admin | propose duplicate groups |
| GET | `/api/admin/export/gold?layer=` | admin | gold-standard JSONL (training-eligible only) |
| GET | `/api/admin/splits/plan?level=object|institution|culture|region` | admin | leakage-safe split preview |
| POST | `/api/admin/references/reload`, `/api/admin/taxonomy/reload` | admin | hot reload |
| GET | `/api/health` | — | versions + index status |

Analysis report schema `global-face-analysis/1.0`: `input{original_file, sha256, phash, working_size, analysis_context{domain, depicted}}`, `versions`, `parameters`, `wording{expression, actions}`, `image_quality`, `face_count`, `faces[]{face_index, box, detector, detection_confidence, quality, pose, facial_actions, expression, reference_retrieval, morphometrics, difficulty_level, stability, artefacts, status}`, `summary`, `status`; `object_metadata` is appended by the API layer after analysis.
