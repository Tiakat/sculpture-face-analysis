"""Report exports for the global platform: JSON (archival), CSV (flat), PDF."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def flatten(d, prefix="", out=None):
    out = {} if out is None else out
    if isinstance(d, dict):
        for k, v in d.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        for i, item in enumerate(d):
            flatten(item, f"{prefix}[{i}]", out)
    else:
        out[prefix] = json.dumps(d) if isinstance(d, (list, dict)) else d
    return out


def to_csv(report: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["field", "value"])
    for k, v in flatten(report).items():
        w.writerow([k, v])
    return buf.getvalue()


def _tbl(rows, widths=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3e6ea")),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]))
    return t


def to_pdf(report: dict, analysis_dir: Path, object_meta: dict | None = None, annotations: list[dict] | None = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm, title="Face Heritage Analysis Report")
    ss = getSampleStyleSheet()
    h1, h2, body = ss["Heading1"], ss["Heading2"], ss["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#444"))
    el = [Paragraph("Global Face Heritage Platform — Analysis Report", h1),
          Paragraph(f"Image analysis ID: <b>{report.get('analysis_id', '—')}</b> · Status: <b>{report['status']}</b> · "
                    f"Generated: {report.get('finished_at', '')}", small),
          Paragraph(report["wording"]["expression"], small), Spacer(1, 8)]

    el.append(Paragraph("1. Object and provenance (kept separate from the facial computation)", h2))
    rows = [["Field", "Value"], ["File", report["input"]["original_file"]], ["SHA-256", report["input"]["sha256"][:32] + "…"],
            ["Perceptual hash", report["input"].get("phash", "")]]
    for k, v in (object_meta or {}).items():
        if v not in (None, "", [], {}):
            rows.append([k, str(v)])
    el.append(_tbl(rows, [4.5 * cm, 12.5 * cm]))
    imgs = [Image(str(analysis_dir / n), width=7.5 * cm, height=6 * cm, kind="proportional") for n in ("working.jpg", "overlay.jpg")
            if (analysis_dir / n).exists()]
    if imgs:
        el += [Spacer(1, 6), Table([imgs]), Paragraph("Working image · face detections and landmarks", small)]

    q = report["image_quality"]
    el.append(Paragraph("2. Image quality", h2))
    el.append(_tbl([["Metric", "Value"], ["Score (0-100)", q["score"]], ["Resolution", f"{q['width']}x{q['height']}"],
                    ["Sharpness", q["blur_laplacian_var"]], ["Mean luminance", q["mean_luminance"]], ["Issues", "; ".join(q["issues"]) or "none"],
                    ["Faces detected", report.get("face_count", 0)]], [6 * cm, 11 * cm]))
    if report["status"] != "completed":
        el.append(Paragraph(f"<b>Not completed:</b> {report.get('rejection_reason', '')}", body))
        doc.build(el)
        return buf.getvalue()

    for f in report["faces"]:
        el.append(PageBreak())
        el.append(Paragraph(f"Face #{f['face_index']}", h1))
        if f.get("status") != "completed":
            el.append(Paragraph(f"Rejected: {f.get('rejection_reason')}", body))
            continue
        fimgs = [Image(str(analysis_dir / f["artefacts"][k]), width=5 * cm, height=5 * cm, kind="proportional")
                 for k in ("crop", "aligned") if (analysis_dir / f["artefacts"][k]).exists()]
        if fimgs:
            el.append(Table([fimgs]))
        p, ex = f["pose"], f["expression"]
        el.append(Paragraph("Geometry", h2))
        el.append(_tbl([["Metric", "Value"], ["Face quality", f["quality"]["score"]], ["Yaw / pitch / roll", f"{p['yaw_deg']} / {p['pitch_deg']} / {p['roll_deg']}"],
                        ["Frontalness", p["frontalness"]], ["Symmetry", p["symmetry_index"]], ["Pose quality", p["pose_quality"]],
                        ["Difficulty level (0-6)", f["difficulty_level"]], ["Detector", f"{f['detector']} ({f['detection_confidence']})"]], [6 * cm, 11 * cm]))
        el.append(Paragraph("Expression", h2))
        el.append(Paragraph(f"Primary: <b>{ex['primary']}</b> · confidence {ex['confidence']}" + (f" · <i>{ex['unknown_reason']}</i>" if ex["is_unknown"] else ""), body))
        el.append(_tbl([["Category", "Probability"]] + [[c, f"{v:.3f}"] for c, v in ex["ranked"]], [6 * cm, 4 * cm]))
        st = (f.get("stability") or {}).get("summary") or {}
        if st:
            el.append(Paragraph(f"Stability score {st.get('stability_score')} · illumination-consistent {st.get('illumination_top_class_consistent')}", small))
        el.append(Paragraph("Facial-action configuration", h2))
        el.append(Paragraph(report["wording"]["actions"], small))
        el.append(_tbl([["AU", "Description", "Intensity", "Evidence", "Associated musculature"]] +
                       [[a["code"], a["name"], a["intensity"], a["evidence"], a["associated_musculature"]] for a in f["facial_actions"]["action_units"]],
                       [1.4 * cm, 3.6 * cm, 1.8 * cm, 1.8 * cm, 8.4 * cm]))
        m = f["morphometrics"]
        el.append(Paragraph("Geometric morphometrics", h2))
        el.append(Paragraph(m["interpretation"], body))
        el.append(_tbl([["Index", "Value"]] + [[k, v] for k, v in m["linear_measurements"].items()] +
                       [["Procrustes distance to consensus", m["procrustes_distance_to_consensus"]], ["Reference n", m["reference_sample_size"]]], [7 * cm, 10 * cm]))
        for fld, aff in m["group_affinity"].items():
            el.append(Paragraph(f"Affinity by <b>{fld}</b> (reliability {aff['reliability']})", body))
            el.append(_tbl([["Group", "n", "Affinity", "Dist. to mean", "Min dist."]] +
                           [[r["group"], r["n"], r["affinity"], r["distance_to_group_mean"], r["min_distance"]] for r in aff["ranking"]],
                           [5 * cm, 1.5 * cm, 2.5 * cm, 3.5 * cm, 3 * cm]))
        for c in m["caveats"]:
            el.append(Paragraph("• " + c, small))
        rr = f["reference_retrieval"]
        if rr["nearest_overall"]:
            el.append(Paragraph("Closest reference specimens", h2))
            el.append(_tbl([["Reference", "Sim.", "Institution", "Culture", "Region", "Date", "Expr."]] +
                           [[r["reference_id"], r["similarity"], r.get("museum") or "", r.get("culture") or "", r.get("region") or "", r.get("date") or "", r.get("expression") or ""]
                            for r in rr["nearest_overall"][:10]], [3 * cm, 1.4 * cm, 3.6 * cm, 3 * cm, 2.6 * cm, 2 * cm, 1.6 * cm]))
        fa = [a for a in (annotations or []) if a.get("face_index") == f["face_index"]]
        if fa:
            el.append(Paragraph("Human annotations", h2))
            el.append(_tbl([["Date", "Annotator", "Layer", "Verdict", "Expression", "Confidence", "Comment"]] +
                           [[a.get("created_at", "")[:19], a.get("annotator", ""), a.get("layer", ""), a.get("verdict", ""), a.get("expression") or "", a.get("confidence") or "", a.get("comment") or ""] for a in fa],
                           [3 * cm, 2.5 * cm, 2 * cm, 1.8 * cm, 2.2 * cm, 1.8 * cm, 3.7 * cm]))

    el.append(PageBreak())
    el.append(Paragraph("Reproducibility", h2))
    el.append(_tbl([["Component", "Version"]] + [[k, v] for k, v in report["versions"].items()] + [["Parameters", json.dumps(report["parameters"])]], [4.5 * cm, 12.5 * cm]))
    doc.build(el)
    return buf.getvalue()
