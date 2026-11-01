# Data model and ontology

```
USER ──creates──▶ OBJECT ──has──▶ IMAGE ──contains──▶ FACE ──has──▶ ANNOTATION (audited)
                    │                │                  │
                    │                └─ sha256, phash,  └─ analysis JSON: landmarks, AUs, expression,
                    │                   view_angle,        morphometrics, retrieval, stability, difficulty
                    │                   image_license,
                    │                   transformations
                    ├─ member_of ──▶ COLLECTION (versioned research subset, e.g. "Moche portrait vessels v0.3")
                    └─ in ──▶ DUPLICATE_GROUP (proposed by sha256 / perceptual hash; reviewed by humans)
```

## OBJECT (table `objects`)
Provenance: `source`, `source_object_id`, `source_url`, `institution`, `object_number`, `title`, `artist`.
Taxonomy (controlled): `domain`, `region`, `culture`, `culture_subgroup`, `period`, `material`, `technique`, `object_type`, `subject_type`, `representation_type`, `restoration_status`, `polychromy`.
Chronology with uncertainty: `date_start`, `date_end`, `date_estimate`, `date_precision`, `date_confidence`, `date_display` (astronomical years; -400 = 401 BCE).
Geography: `country`, `site`, `provenance_text`. Grouping: `mold_group`, `duplicate_group_id`.
Rights/ethics: `license`, `rights_holder`, `training_allowed` (derived), `cultural_sensitivity[]`, `access_restriction`.
Knowledge links: `external_ids` (Wikidata QID, Getty AAT/TGN/ULAN, PeriodO), `literature[]`, `notes`, `annotation_layer`, `metadata_version`.

## IMAGE (`images`)
`original_filename`, `stored_path` (immutable), `sha256`, `phash`, `width`, `height`, `image_url`, `iiif_url`, `photographer`, `image_license` (may differ from object metadata licence), `view_angle`, `transformations[]` (processing provenance), `is_synthetic`, `derived_from_image_id`, analysis summary (`status`, `quality_score`, `face_count`, `model_version`, `dataset_version`).

## FACE (`faces`)
`face_index`, `bbox`, `detection_confidence`, `quality_score`, `pose_quality`, `yaw`, `expression`, `expression_confidence`, `difficulty_level` (0–6), `face_completeness`, `flagged`, `status`. Full numeric results live in `data/analyses/<image_id>/analysis.json` → `faces[face_index]` and `face_<i>/landmarks_478.npy`, `embedding.npy`.

## ANNOTATION (`annotations`)
`layer` (gold_standard | expert), `verdict`, `expression`, `expression_distribution` (soft labels), `aus` {AU: present|absent|uncertain}, `face_completeness`, `structures_visible`, `representation_type`, `subject_type`, `difficulty_level`, `confidence`, `comment`, `flag_for_review`, `review_status`, `reviewed_by`, `reviewed_at`, `annotator_id`, `created_at`. Inter-rater agreement is computed from multiple annotations of the same face (κ / α in `training/evaluate.py` extension point).

## Taxonomy (`config/taxonomy.json`)
`domain` (with `depicted` flag and `restricted` flag), `region` (hierarchical, `parent`/`children`), `culture` (with `region`, `date_start`, `date_end`, `subgroups`, `external_ids`), `period`, `material`, `technique`, `object_type`, `subject_type`, `representation_type`, `sex_gender_depicted`, `age_depicted`, `face_completeness`, `restoration_status`, `polychromy`, `view_angle`, `license` (with `training_allowed`, `derivatives`, `attribution`), `cultural_sensitivity`, `date_precision`, `annotation_layer`, `source_institutions`.

## Corpus record (`data/corpus/records.jsonl`)
Canonical schema shared by all connectors: see `scripts/connectors/base.py: CorpusRecord`. `culture_raw` / `material_raw` / `license_raw` keep the institution's original strings; normalised ids are derived and can be re-derived when the alias tables improve. `rights.jsonl` carries the per-image policy. `VERSION` is bumped at every ingestion and stamped into every analysis report.
