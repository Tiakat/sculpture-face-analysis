# Literature map

Grouped by the module each work informs. Items marked ★ are the core references; those in the "from Millones' bibliography" blocks are the references inside the article you supplied that carry method or data the platform reuses. Where I could not verify a DOI I give title + venue only — please check before citing.

## A. Directly comparable computational studies of ancient sculpture ★

1. **Cowen, A. S. & Keltner, D. (2020).** *Universal facial expressions uncovered in art of the ancient Americas: A computational approach.* Science Advances 6(34), eaba1590. — 63 ancient American sculptures, 30 emotion categories, principal preserved components + t-SNE. Central precedent for analysing *depicted* expressions statistically. (PMC7438103)
2. **Tian, F. et al. (2021).** *Uncover Common Facial Expressions in Terracotta Warriors: A Deep Learning Approach.* arXiv:2105.04826. — 58 usable sculpture images; models trained on modern faces transferred poorly; GAN-based expression transfer for augmentation. The key warning for our domain-transfer problem.
3. **Sheng, Y. et al. (2024).** *Facial Recognition and Classification of Terracotta Warriors in the Mausoleum of the First Emperor Using Deep Learning.* ISPRS Annals X-1-2024, 205–212. — 295 faces, enhanced SqueezeNet/FaceNet embedding + agglomerative clustering (our retrieval/clustering analogue).
4. **Millones, M.** *Vasijas-Retrato Moche: Ordenando Rostros (Forma, Tiempo y Espacio).* — Museo Larco photographs, APS 2.42 landmarking, PAST Procrustes + PCA + Ward clustering; realism, individuality, Larco phases I–IV and valleys (Virú 104, Chicama 44, Santa 38, Moche 11…). Our morphometrics module follows it step by step.
5. **Srinivasan, R., Rudolph, C. & Roy-Chowdhury, A. K. (2015).** *Computerized face recognition in Renaissance portrait art.* IEEE Signal Processing Magazine 32(4). — face matching across painted portraits; methodology for identity comparison in art (relevant to Donnan's "same individual on several vessels").
6. **Achlioptas, P. et al. (2021).** *ArtEmis: Affective Language for Visual Art.* CVPR. — 450k emotion annotations on WikiArt; large-scale affect-from-art baseline (paintings, not sculpture).

## B. Moche portrait vessels — archaeology and art history

7. **Donnan, C. B. (2004).** *Moche Portraits from Ancient Peru.* University of Texas Press. — the corpus study; same individual as ruler and as prisoner; juvenile→adult sequences; scars as identifiers. Basis for identity-level retrieval and for the "mould/individual" leakage rule.
8. **Larco Hoyle, R. (1948).** *Cronología arqueológica del norte del Perú.* / **(2001 [1939])** *Los Mochicas.* — the five-phase stirrup-spout chronology used as the `phase` field.
9. **Bawden, G. (1994).** *La paradoja estructural: la cultura Moche como ideología política.* In Uceda & Mujica (eds.) *Moche: Propuestas y Perspectivas.* — portraits restricted to Chicama/Moche/Virú and phases III–IV; individuality as a political device.
10. **Hocquenghem, A. M. (1977).** *Un "vase portrait" de femme mochica.* Ñawpa Pacha 15. — the female portrait vessel; braids as sex criterion; shaman interpretation.
11. **Verano, J. W. (1994).** *Características físicas y biológicas de los Moche.* In *Moche: Propuestas y Perspectivas.* — skeletal description ("wide faces, narrow noses") to compare with sculpted morphology.
12. **Millaire, J.-F. (2002).** *Moche Burial Patterns.* BAR International Series 1066. — burial context of portrait vessels.
13. **Vergara, E. M. (1992).** *La pintura facial en la cerámica mochica: catálogo preliminar.* Revista del Museo de Arqueología UNT 3. — facial paint catalogue (future metadata field: face paint may bias appearance embeddings).
14. **Villanueva, M. (2003).** *Forma y fenotipo facial.* Estudios de Antropología Biológica XI. — Pöch facial-form typology; a qualitative counterpart to the PCA axes.
15. **Quilter, J. (2010).** *The Moche of Ancient Peru: Media and Messages.* Peabody Museum Press; **Jackson, M. A. (2008).** *Moche Art and Visual Culture in Ancient Peru.* UNM Press. — context for the "individual vs social type" debate.

## C. FACS, facial actions and the limits of emotion inference ★

16. **Ekman, P. & Friesen, W. V. (1978); Ekman, Friesen & Hager (2002).** *Facial Action Coding System.* — the AU inventory (≈44–46 AUs; >7,000 observed combinations). FACS is descriptive; emotion prediction comes from EMFACS tables.
17. **Ekman, P. & Rosenberg, E. (eds.) (1997).** *What the Face Reveals.* OUP. — applied FACS; the source Millones cites for selecting *neutral* vessels — the same criterion our calibration script uses.
18. **Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M. & Pollak, S. D. (2019).** *Emotional expressions reconsidered: Challenges to inferring emotion from human facial movements.* Psychological Science in the Public Interest 20(1). — the definitive caution against reading emotions from faces; motivates "depicted expression" wording and the UNKNOWN class.
19. **Baltrušaitis, T., Zadeh, A., Lim, Y. C. & Morency, L.-P. (2018).** *OpenFace 2.0: Facial Behavior Analysis Toolkit.* IEEE FG. — open AU intensity estimation (0–5) on photographs; a candidate replacement for our geometric AU estimator once installed (C++ build).
20. **Cheong, J. H. et al. (2023).** *Py-Feat: Python Facial Expression Analysis Toolbox.* Affective Science. — Python AU/emotion detectors; alternative backend.
21. **Kubovy, M. (1992).** *Caricature and face recognition.* Memory & Cognition 20(4). — exaggeration improves recognition; explains why stylised vessels may still individualise (cited by Millones).

## D. Geometric morphometrics

22. **Slice, D. E. (ed.) (2005).** *Modern Morphometrics in Physical Anthropology.* Kluwer/Plenum. — Millones' methodological source.
23. **Rohlf, F. J. & Slice, D. E. (1990).** *Extensions of the Procrustes method for the optimal superimposition of landmarks.* Systematic Zoology 39. — GPA as implemented in `morphometrics.gpa`.
24. **Bookstein, F. L. (1991).** *Morphometric Tools for Landmark Data.* CUP. — shape coordinates, PCA of shape.
25. **Adams, D. C., Rohlf, F. J. & Slice, D. E. (2013).** *A field comes of age: geometric morphometrics in the 21st century.* Hystrix 24(1). — review.
26. **Klingenberg, C. P. (2011).** *MorphoJ.* Molecular Ecology Resources 11. — reference software for cross-checking our Procrustes/PCA output.
27. **Hammer, Ø., Harper, D. & Ryan, P. (2001).** *PAST: Paleontological Statistics software package.* Palaeontologia Electronica 4(1). — the software Millones used; export our Procrustes coordinates to PAST to reproduce his plots.
28. **Enlow, D. H. & Hans, M. G. (1996).** *Essentials of Facial Growth.* Saunders. — facial asymmetry as biological reality (Millones).
29. **Prag, J. & Neave, R. (1997).** *Making Faces: Using Forensic and Archaeological Evidence.* British Museum Press. — forensic reconstruction; the soft-tissue gap that portrait vessels might inform.
30. **Tiesler, V., Cucina, A. & Romano, A. (2004)** and **Kustár, Á. (2004)**, HOMO 55 — skull-to-portrait comparisons (Red Queen of Palenque; Antal Simon).

## E. Face landmarking, pose, quality

31. **Kartynnik, Y., Ablavatski, A., Grishchenko, I. & Grundmann, M. (2019).** *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs.* CVPR Workshops. — the FaceMesh model (468 landmarks) we use.
32. **Bazarevsky, V. et al. (2019).** *BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs.* — our detector; also the reason the same models run on Android/iOS later.
33. **Lugaresi, C. et al. (2019).** *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172.
34. **Pech-Pacheco, J. L. et al. (2000).** *Diatom autofocusing in brightfield microscopy: a comparative study.* ICPR — Laplacian-variance sharpness measure used by the quality gate.
35. **Zuiderveld, K. (1994).** *Contrast Limited Adaptive Histogram Equalization.* Graphics Gems IV. — CLAHE.
36. **Dalal, N. & Triggs, B. (2005).** *Histograms of Oriented Gradients for Human Detection.* CVPR. — HOG appearance component of the embedding.

## F. Human facial-expression datasets (pre-training / benchmarking only)

37. **Gross, R. et al. (2010).** *Multi-PIE.* Image and Vision Computing 28(5). — 337 subjects, 15 viewpoints, 19 illuminations.
38. **Langner, O. et al. (2010).** *Presentation and validation of the Radboud Faces Database.* Cognition & Emotion 24(8). — 8 expressions, 5 camera angles.
39. **Savran, A. et al. (2008).** *Bosphorus Database for 3D Face Analysis.* BIOID. — 3-D, AU-coded, pose and occlusion.
40. **Cosker, D., Krumhuber, E. & Hilton, A. (2011).** *A FACS valid 3D dynamic action unit database with applications to 3D dynamic morphable facial modeling.* ICCV. — D3DFACS.
41. **Namba, S. et al. (2023).** *Development of the RIKEN database for dynamic facial expressions with multiple angles.* Scientific Reports. — multi-angle, FACS-coded.
42. **Lucey, P. et al. (2010).** *The Extended Cohn-Kanade Dataset (CK+).* CVPRW. **Mavadati, S. M. et al. (2013).** *DISFA.* IEEE TAC. **Zhang, X. et al. (2014).** *BP4D-Spontaneous.* Image and Vision Computing. — AU-intensity training sets.
43. **Mollahosseini, A., Hasani, B. & Mahoor, M. H. (2019).** *AffectNet.* IEEE TAC 10(1). — 1M in-the-wild faces (categorical + valence/arousal).
44. **Li, S. & Deng, W. (2020).** *Deep Facial Expression Recognition: A Survey.* IEEE TAC. — landscape of FER methods and their known biases.

## G. Methodological hygiene

45. **Kapoor, S. & Narayanan, A. (2023).** *Leakage and the reproducibility crisis in machine-learning-based science.* Patterns 4(9). — why `training/split.py` splits by object/mould.
46. **Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017).** *On Calibration of Modern Neural Networks.* ICML. — expected calibration error used in `training/evaluate.py`.
47. **Mitchell, M. et al. (2019).** *Model Cards for Model Reporting.* FAT*. — template for documenting each model version in `data/models/registry.json`.

## Suggested search strings for extending this list

`"portrait vessel" Moche morphometric` · `"huaco retrato" análisis` · `sculpture "facial expression" deep learning heritage` · `"geometric morphometrics" face sculpture` · `"action units" statue OR sculpture OR artwork` · `terracotta warriors face clustering` · `Cowen Keltner ancient sculpture replication`
