"""Connector framework for the Global Face Corpus.

Every connector yields `CorpusRecord`s in ONE canonical schema (below), so
the corpus, the reference index and the platform database never depend on a
museum's field names.  Images are fetched through IIIF where the institution
supports it (Europeana, Getty, Rijksmuseum, ARTIC, Smithsonian, Harvard).

Corpus layout (data/corpus/):
  images/<SOURCE>_<ID>/original.<ext>, thumb.jpg
  records.jsonl        canonical records
  rights.jsonl         licence / attribution / training policy per image
  VERSION, CHANGELOG.md

Rights rule: only CC0 / public-domain / CC BY / CC BY-SA images are stored by
the automatic connectors.  NC/ND/unknown are recorded as *metadata-only*
(no image) so the object is still known to the knowledge base.
"""
from __future__ import annotations

import os
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app import taxonomy  # noqa: E402

CORPUS = Path(os.environ.get("GFP_DATA_DIR", ROOT / "data")) / "corpus"
IMG_DIR = CORPUS / "images"
RECORDS = CORPUS / "records.jsonl"
RIGHTS = CORPUS / "rights.jsonl"
HEADERS = {"User-Agent": "GlobalFaceHeritagePlatform/0.1 (academic research)"}

FACE_WORDS = ("portrait", "head", "bust", "face", "mask", "figure", "figurine", "statue", "sculpture", "effigy",
              "self-portrait", "retrato", "cabeza", "rostro", "busto", "tête", "visage", "kopf", "porträt", "maske")
NOT_FACE_WORDS = ("landscape", "still life", "vase without", "textile fragment", "coin die", "architectural")


@dataclass
class CorpusRecord:
    id: str
    source: str
    source_object_id: str
    source_url: str | None
    institution: str | None
    object_number: str | None
    title: str | None
    artist: str | None = None
    domain: str = "sculpture"
    region: str | None = None
    culture: str | None = None
    culture_raw: str | None = None          # institution's original culture string (never used by the model)
    period: str | None = None
    date_start: int | None = None
    date_end: int | None = None
    date_display: str | None = None
    date_precision: str | None = None
    material: str | None = None
    material_raw: str | None = None
    technique: str | None = None
    object_type: str | None = None
    country: str | None = None
    site: str | None = None
    provenance: str | None = None
    license: str = "UNKNOWN"
    license_raw: str | None = None
    rights_holder: str | None = None
    image_url: str | None = None
    iiif_url: str | None = None
    additional_images: list = field(default_factory=list)
    image_path: str | None = None
    thumbnail_path: str | None = None
    image_only_metadata: bool = False       # True when rights forbid storing the image
    external_ids: dict = field(default_factory=dict)
    expression: str | None = None           # ONLY expert labels go here — never museum text
    expression_source: str | None = None
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ normalisers
LICENSE_MAP = [
    (re.compile(r"cc0|publicdomain/zero|public domain dedication", re.I), "CC0"),
    (re.compile(r"public domain|publicdomain/mark|pd-|no known copyright", re.I), "PD"),
    (re.compile(r"by-nc-nd", re.I), "CC-BY-NC-ND"),
    (re.compile(r"by-nc", re.I), "CC-BY-NC"),
    (re.compile(r"by-sa", re.I), "CC-BY-SA"),
    (re.compile(r"cc[- ]by|creativecommons\.org/licenses/by/", re.I), "CC-BY"),
    (re.compile(r"in copyright|rights reserved|InC", re.I), "RESTRICTED"),
]


def normalise_license(raw: str | None, is_public_domain: bool | None = None) -> str:
    if is_public_domain:
        return "PD"
    if not raw:
        return "UNKNOWN"
    for rx, lic in LICENSE_MAP:
        if rx.search(raw):
            return lic
    return "UNKNOWN"


def storable(license_id: str) -> bool:
    return license_id in ("CC0", "PD", "CC-BY", "CC-BY-SA")


CULTURE_ALIASES = {
    "moche": "moche", "mochica": "moche", "viru": "viru_gallinazo", "virú": "viru_gallinazo", "gallinazo": "viru_gallinazo",
    "recuay": "recuay", "nasca": "nasca", "nazca": "nasca", "wari": "wari", "huari": "wari", "tiwanaku": "tiwanaku",
    "lambayeque": "lambayeque_sican", "sican": "lambayeque_sican", "sicán": "lambayeque_sican", "chimu": "chimu", "chimú": "chimu",
    "inca": "inca", "inka": "inca", "chavin": "chavin", "chavín": "chavin", "olmec": "olmec", "maya": "maya", "teotihuacan": "teotihuacan",
    "aztec": "aztec", "mexica": "aztec", "colima": "west_mexico", "jalisco": "west_mexico", "nayarit": "west_mexico",
    "egypt": "egyptian", "egyptian": "egyptian", "nubia": "nubian_kushite", "kush": "nubian_kushite", "meroitic": "nubian_kushite",
    "nok": "nok", "ife": "ife", "benin": "benin", "edo": "benin", "yoruba": "yoruba", "kongo": "kongo",
    "sumer": "sumerian_akkadian", "akkad": "sumerian_akkadian", "babylon": "babylonian_assyrian", "assyria": "babylonian_assyrian",
    "achaemenid": "achaemenid", "persian": "achaemenid", "cycladic": "cycladic_minoan_mycenaean", "minoan": "cycladic_minoan_mycenaean",
    "mycenaean": "cycladic_minoan_mycenaean", "archaic greek": "greek_archaic", "classical greek": "greek_classical",
    "hellenistic": "greek_hellenistic", "greek": "greek_classical", "etruscan": "etruscan", "roman": "roman_imperial",
    "republican": "roman_republican", "byzantine": "byzantine", "celtic": "celtic_iron_age", "romanesque": "medieval_romanesque",
    "gothic": "medieval_gothic", "renaissance": "renaissance", "baroque": "baroque", "rococo": "baroque", "neoclassic": "neoclassical",
    "indus": "indus", "harappa": "indus", "gandhara": "gandhara", "gupta": "gupta_mathura", "mathura": "gupta_mathura",
    "chola": "chola", "buddhist": "buddhist_traditions", "qin": "chinese_qin_han", "han dynasty": "chinese_qin_han",
    "tang": "chinese_tang", "song dynasty": "chinese_song_ming_qing", "ming": "chinese_song_ming_qing", "qing": "chinese_song_ming_qing",
    "jomon": "japanese_jomon_kofun", "jōmon": "japanese_jomon_kofun", "kofun": "japanese_jomon_kofun", "haniwa": "japanese_jomon_kofun",
    "nara": "japanese_buddhist_edo", "kamakura": "japanese_buddhist_edo", "edo period": "japanese_buddhist_edo", "korea": "korean",
    "khmer": "khmer", "angkor": "khmer", "java": "javanese", "sepik": "papua_sepik", "maori": "maori", "māori": "maori",
    "rapa nui": "rapa_nui", "easter island": "rapa_nui", "hawai": "hawaiian", "inuit": "inuit_yupik", "yup'ik": "inuit_yupik",
    "haida": "northwest_coast", "tlingit": "northwest_coast", "kwakwaka": "northwest_coast",
}
MATERIAL_ALIASES = [("terracotta", "terracotta"), ("ceramic", "ceramic"), ("earthenware", "ceramic"), ("clay", "ceramic"),
                    ("marble", "stone_marble"), ("limestone", "stone_limestone"), ("basalt", "stone_basalt_volcanic"),
                    ("andesite", "stone_basalt_volcanic"), ("sandstone", "stone_other"), ("granite", "stone_other"), ("stone", "stone_other"),
                    ("bronze", "metal_bronze"), ("gold", "metal_gold_silver"), ("silver", "metal_gold_silver"), ("copper", "metal_other"),
                    ("iron", "metal_other"), ("wood", "wood"), ("ivory", "ivory_bone_shell"), ("bone", "ivory_bone_shell"), ("shell", "ivory_bone_shell"),
                    ("plaster", "plaster_stucco"), ("stucco", "plaster_stucco"), ("wax", "wax"), ("faience", "glass_faience"), ("glass", "glass_faience"),
                    ("oil on canvas", "canvas"), ("canvas", "canvas"), ("panel", "panel"), ("fresco", "wall_fresco"), ("mosaic", "mosaic"),
                    ("paper", "paper_parchment"), ("parchment", "paper_parchment"), ("vellum", "paper_parchment"),
                    ("gelatin silver", "photographic_print"), ("albumen", "photographic_print"), ("daguerreotype", "photographic_print")]
DOMAIN_HINTS = [("photograph", "photograph_object"), ("daguerreotype", "photograph_object"), ("gelatin silver", "photograph_object"),
                ("painting", "painting"), ("oil on", "painting"), ("tempera", "painting"), ("fresco", "painting"), ("mosaic", "painting"),
                ("drawing", "drawing"), ("sketch", "drawing"), ("watercolor", "drawing"), ("watercolour", "drawing"),
                ("engraving", "print"), ("etching", "print"), ("lithograph", "print"), ("woodcut", "print"), ("print", "print"),
                ("manuscript", "manuscript"), ("illumination", "manuscript"), ("mask", "mask"), ("relief", "relief"), ("stela", "relief"),
                ("coin", "coin_medal"), ("medal", "coin_medal"), ("seal", "coin_medal"), ("vessel", "ceramic_figure"), ("bottle", "ceramic_figure"),
                ("jar", "ceramic_figure"), ("figurine", "sculpture"), ("statue", "sculpture"), ("bust", "sculpture"), ("head", "sculpture")]
OBJECT_TYPE_HINTS = [("portrait head", "portrait_head"), ("head", "portrait_head"), ("bust", "bust"), ("mask", "mask"), ("relief", "relief"),
                     ("stela", "stela"), ("figurine", "statuette_figurine"), ("statuette", "statuette_figurine"), ("statue", "statue"),
                     ("figure", "full_figure"), ("vessel", "vessel_effigy"), ("bottle", "vessel_effigy"), ("coin", "coin_medal"), ("medal", "coin_medal"),
                     ("self-portrait", "self_portrait"), ("portrait", "portrait_painting"), ("group", "group_painting")]


def normalise_culture(*texts: str | None) -> str | None:
    blob = " ".join(t for t in texts if t).lower()
    for key in sorted(CULTURE_ALIASES, key=len, reverse=True):
        if key in blob:
            return CULTURE_ALIASES[key]
    return None


def normalise_material(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.lower()
    for key, val in MATERIAL_ALIASES:
        if key in r:
            return val
    return "unknown"


def guess_domain(*texts: str | None, default="sculpture") -> str:
    blob = " ".join(t for t in texts if t).lower()
    for key, val in DOMAIN_HINTS:
        if key in blob:
            return val
    return default


def guess_object_type(*texts: str | None) -> str | None:
    blob = " ".join(t for t in texts if t).lower()
    for key, val in OBJECT_TYPE_HINTS:
        if key in blob:
            return val
    return None


def period_from_dates(ds: int | None, de: int | None) -> str | None:
    if ds is None and de is None:
        return None
    mid = ((ds if ds is not None else de) + (de if de is not None else ds)) / 2
    if mid < -3500:
        return "prehistoric"
    if mid < 500:
        return "ancient"
    if mid < 1500:
        return "medieval"
    if mid < 1800:
        return "early_modern"
    if mid < 1970:
        return "modern"
    return "contemporary"


def region_from_culture(culture: str | None) -> str | None:
    t = taxonomy.term("culture", culture) if culture else None
    return t.get("region") if t else None


def looks_like_face_object(*texts: str | None) -> bool:
    blob = " ".join(t for t in texts if t).lower()
    return any(w in blob for w in FACE_WORDS) and not any(w in blob for w in NOT_FACE_WORDS)


# ------------------------------------------------------------------ IO helpers
def client() -> httpx.Client:
    return httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True)


def get_json(c: httpx.Client, url: str, params=None, retries: int = 3):
    for i in range(retries):
        try:
            r = c.get(url, params=params)
            if r.status_code in (429, 403):        # throttled by the museum API: back off and retry
                time.sleep(6 * (i + 1))
                if i < retries - 1:
                    continue
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            if i == retries - 1:
                code = getattr(getattr(e, "response", None), "status_code", None)
                why = {404: "record no longer exists", 403: "museum server refused this one"}.get(code, f"{code or 'network'}")
                print(f"  - skipped ({why}): {url.rsplit('/', 1)[-1][:60]}")
                return None
            time.sleep(2 * (i + 1))
    return None


def iiif_image_url(base: str, width: int = 1600) -> str:
    """Standard IIIF Image API request for a full image resized to `width` px."""
    return f"{base.rstrip('/')}/full/{width},/0/default.jpg"


def existing_ids() -> set[str]:
    if not RECORDS.exists():
        return set()
    return {json.loads(l)["id"] for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()}


def save_image(c: httpx.Client, url: str, record_id: str) -> tuple[str, str] | None:
    d = IMG_DIR / record_id
    d.mkdir(parents=True, exist_ok=True)
    try:
        r = c.get(url)
        r.raise_for_status()
    except httpx.HTTPError:
        print(f"  - image not downloadable: {url[:80]}")
        return None
    ext = Path(url.split("?")[0]).suffix.lower()
    ext = ext if ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp") else ".jpg"
    orig = d / f"original{ext}"
    orig.write_bytes(r.content)
    try:
        im = Image.open(orig).convert("RGB")
        im.thumbnail((400, 400))
        im.save(d / "thumb.jpg", quality=88)
    except Exception as e:
        print(f"  ! not an image: {e}", file=sys.stderr)
        orig.unlink(missing_ok=True)
        return None
    return str(orig.relative_to(CORPUS)), str((d / "thumb.jpg").relative_to(CORPUS))


def store(c: httpx.Client, rec: CorpusRecord, attribution: str | None) -> bool:
    """Download image if rights allow, then append the record + rights entry."""
    if rec.culture and not rec.region:
        rec.region = region_from_culture(rec.culture)
    if rec.period is None:
        rec.period = period_from_dates(rec.date_start, rec.date_end)
    if storable(rec.license) and (rec.iiif_url or rec.image_url):
        url = iiif_image_url(rec.iiif_url) if rec.iiif_url else rec.image_url
        paths = save_image(c, url, rec.id)
        if not paths:
            return False
        rec.image_path, rec.thumbnail_path = paths
    else:
        rec.image_only_metadata = True
    CORPUS.mkdir(parents=True, exist_ok=True)
    with RECORDS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    pol = taxonomy.license_policy(rec.license)
    with RIGHTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": rec.id, "license": rec.license, "license_raw": rec.license_raw, "attribution": attribution,
                            "rights_holder": rec.rights_holder, "source_url": rec.source_url, "image_url": rec.image_url,
                            "image_stored": not rec.image_only_metadata, **pol}, ensure_ascii=False) + "\n")
    return True


def bump_version(note: str) -> str:
    vf = CORPUS / "VERSION"
    n = 0
    if vf.exists():
        try:
            n = int(vf.read_text(encoding="utf-8").strip().split(".")[-1])
        except ValueError:
            pass
    ver = f"GlobalFaceCorpus v0.{n + 1}"
    vf.write_text(ver + "\n", encoding="utf-8")
    with (CORPUS / "CHANGELOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {ver} — {datetime.now(timezone.utc).date()} — {note}\n")
    print(f"Corpus stamped {ver}")
    return ver
