"""Institution connectors.  Each `ingest_<source>(queries, limit, **opts)` yields nothing
and writes canonical records via base.store().  API field names are confined here.

Keys (environment variables): EUROPEANA_API_KEY, SMITHSONIAN_API_KEY (api.data.gov),
HARVARD_API_KEY.  Met, ARTIC, Getty, Rijksmuseum (search), Wikimedia need none.
"""
from __future__ import annotations

import os
import re
import time

from .base import (CorpusRecord, bump_version, client, existing_ids, get_json, guess_domain, guess_object_type,
                   looks_like_face_object, normalise_culture, normalise_license, normalise_material, store)

# Broad, multi-cultural default search terms so that one click builds a large, diverse corpus.
DEFAULT_QUERIES = [
    "portrait head", "portrait bust", "portrait sculpture", "head of a man", "head of a woman", "mask", "figurine head",
    "moche portrait", "mochica head", "nasca figure", "chimu figure", "maya figurine", "aztec head", "olmec head", "colima figure",
    "roman portrait", "roman bust", "greek head", "hellenistic head", "etruscan head", "cycladic head",
    "egyptian head", "pharaoh head", "fayum portrait", "nubian head", "mesopotamian head", "sumerian statue",
    "benin head", "ife head", "yoruba mask", "nok head", "kongo figure", "african mask", "fang reliquary head", "dan mask",
    "baule figure", "luba figure", "senufo mask", "dogon figure", "bamana mask", "chokwe mask", "kuba mask", "lega mask",
    "sapi head", "akan head", "ashanti figure", "songye figure", "mende helmet mask", "punu mask", "igbo mask",
    "oceanic mask", "maori figure", "papua new guinea mask", "easter island figure", "inuit mask", "northwest coast mask",
    "buddha head", "bodhisattva head", "gandhara head", "khmer head", "terracotta head", "haniwa", "chinese figure head",
    "japanese mask", "noh mask", "korean figure", "indian head sculpture",
    "marble bust", "bronze head", "wooden mask", "stone head", "ivory head",
    "portrait painting", "self-portrait", "portrait of a man", "portrait of a woman", "portrait drawing", "portrait print",
    "portrait photograph", "daguerreotype portrait", "medieval head", "gothic head", "renaissance bust", "baroque bust",
]


def _year(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        m = re.search(r"-?\d{1,4}", str(s or ""))
        return int(m.group()) if m else None


def met_search(c, q: str, max_ids: int = 1000) -> list[int]:
    """Met Collection API search.  /v1/search (one response with every ID) was retired on
    2026-10-01; /v1.1/search is paginated with offset/limit.  Falls back to v1 for older mirrors."""
    base = "https://collectionapi.metmuseum.org/public/collection"
    ids: list[int] = []
    offset, page = 0, 100
    while len(ids) < max_ids:
        res = get_json(c, f"{base}/v1.1/search", {"q": q, "hasImages": "true", "limit": page, "offset": offset})
        batch = (res or {}).get("objectIDs") or []
        if not batch:
            break
        ids += batch
        offset += len(batch)
        if offset >= int((res or {}).get("total") or 0):
            break
    if not ids and offset == 0:
        res = get_json(c, f"{base}/v1/search", {"q": q, "hasImages": "true"})
        ids = (res or {}).get("objectIDs") or []
    return ids[:max_ids]


# ------------------------------------------------------------------ The Met
def ingest_met(queries=None, limit=500, sleep=0.6, culture_filter: str | None = None):
    api = "https://collectionapi.metmuseum.org/public/collection/v1"
    seen = existing_ids()
    ids: list[int] = []
    n = 0
    with client() as c:
        for q in (queries or DEFAULT_QUERIES):
            for i in met_search(c, q):
                if i not in ids:
                    ids.append(i)
        print(f"[MET] {len(ids)} candidates")
        for oid in ids[:limit]:
            rid = f"MET_{oid}"
            if rid in seen:
                continue
            o = get_json(c, f"{api}/objects/{oid}")
            time.sleep(sleep)
            if not o or not o.get("primaryImage"):
                continue
            blob = [o.get("title"), o.get("objectName"), o.get("classification"), o.get("culture")]
            if not looks_like_face_object(*blob):
                continue
            if culture_filter and culture_filter.lower() not in (o.get("culture") or "").lower():
                continue
            rec = CorpusRecord(
                id=rid, source="MET", source_object_id=str(oid), source_url=o.get("objectURL"), institution="The Metropolitan Museum of Art",
                object_number=o.get("accessionNumber"), title=o.get("title"), artist=o.get("artistDisplayName") or None,
                domain=guess_domain(o.get("objectName"), o.get("classification"), o.get("medium")),
                culture=normalise_culture(o.get("culture"), o.get("period"), o.get("title")), culture_raw=o.get("culture"),
                date_start=o.get("objectBeginDate"), date_end=o.get("objectEndDate"), date_display=o.get("objectDate"),
                material=normalise_material(o.get("medium")), material_raw=o.get("medium"), object_type=guess_object_type(o.get("objectName"), o.get("title")),
                country=o.get("country") or None, site=(o.get("locale") or o.get("excavation") or None), provenance=o.get("creditLine"),
                license=normalise_license(o.get("rightsAndReproduction"), o.get("isPublicDomain")), license_raw=o.get("rightsAndReproduction") or ("Open Access CC0" if o.get("isPublicDomain") else None),
                image_url=o.get("primaryImage"), additional_images=o.get("additionalImages") or [],
                external_ids={"met_object_id": oid, "wikidata": o.get("objectWikidata_URL")})
            if store(c, rec, o.get("creditLine")):
                n += 1
                print(f"  + {rid} {rec.title} [{rec.license}]")
    if n:
        bump_version(f"MET: +{n}")
    return n


# ------------------------------------------------------------------ Art Institute of Chicago
def ingest_artic(queries=None, limit=500, sleep=0.3):
    api = "https://api.artic.edu/api/v1/artworks/search"
    iiif = "https://www.artic.edu/iiif/2/{image_id}"
    fields = "id,title,image_id,is_public_domain,date_display,date_start,date_end,place_of_origin,medium_display,artist_display,classification_titles,credit_line,main_reference_number,alt_image_ids,artwork_type_title"
    seen = existing_ids()
    n = 0
    with client() as c:
        for q in (queries or DEFAULT_QUERIES):
            page = 1
            got = 0
            while got < limit:
                res = get_json(c, api, {"q": q, "fields": fields, "limit": 50, "page": page})
                if not res or not res.get("data"):
                    break
                for o in res["data"]:
                    got += 1
                    rid = f"ARTIC_{o['id']}"
                    if rid in seen or not o.get("image_id"):
                        continue
                    if not looks_like_face_object(o.get("title"), o.get("artwork_type_title"), " ".join(o.get("classification_titles") or [])):
                        continue
                    seen.add(rid)
                    rec = CorpusRecord(
                        id=rid, source="ARTIC", source_object_id=str(o["id"]), source_url=f"https://www.artic.edu/artworks/{o['id']}",
                        institution="Art Institute of Chicago", object_number=o.get("main_reference_number"), title=o.get("title"),
                        artist=o.get("artist_display"), domain=guess_domain(o.get("artwork_type_title"), o.get("medium_display")),
                        culture=normalise_culture(o.get("artist_display"), o.get("place_of_origin"), o.get("title")), culture_raw=o.get("artist_display"),
                        date_start=o.get("date_start"), date_end=o.get("date_end"), date_display=o.get("date_display"),
                        material=normalise_material(o.get("medium_display")), material_raw=o.get("medium_display"),
                        object_type=guess_object_type(o.get("title"), o.get("artwork_type_title")), country=o.get("place_of_origin"),
                        provenance=o.get("credit_line"), license="PD" if o.get("is_public_domain") else "RESTRICTED",
                        license_raw="public domain" if o.get("is_public_domain") else "rights reserved",
                        iiif_url=iiif.format(image_id=o["image_id"]), additional_images=[iiif.format(image_id=i) for i in (o.get("alt_image_ids") or [])])
                    if store(c, rec, o.get("credit_line")):
                        n += 1
                        print(f"  + {rid} {rec.title} [{rec.license}]")
                    time.sleep(sleep)
                if not res.get("pagination", {}).get("next_url"):
                    break
                page += 1
    if n:
        bump_version(f"ARTIC: +{n}")
    return n


# ------------------------------------------------------------------ Europeana
def ingest_europeana(queries=None, limit=500, sleep=0.2):
    key = os.getenv("EUROPEANA_API_KEY")
    if not key:
        print("[EUROPEANA] set EUROPEANA_API_KEY (free at https://pro.europeana.eu/page/get-api)")
        return 0
    api = "https://api.europeana.eu/record/v2/search.json"
    seen = existing_ids()
    n = 0
    with client() as c:
        for q in (queries or DEFAULT_QUERIES):
            cursor = "*"
            got = 0
            while got < limit and cursor:
                res = get_json(c, api, {"wskey": key, "query": q, "qf": ["TYPE:IMAGE", "REUSABILITY:open"], "rows": 100,
                                        "cursor": cursor, "profile": "rich"})
                if not res or not res.get("items"):
                    break
                for it in res["items"]:
                    got += 1
                    rid = "EUROPEANA_" + re.sub(r"[^A-Za-z0-9]+", "_", it["id"]).strip("_")
                    if rid in seen:
                        continue
                    title = (it.get("title") or [None])[0]
                    desc = " ".join(it.get("dcDescription") or [])
                    if not looks_like_face_object(title, desc, " ".join(it.get("dcType") or [])):
                        continue
                    seen.add(rid)
                    year = (it.get("year") or [None])[0]
                    rec = CorpusRecord(
                        id=rid, source="EUROPEANA", source_object_id=it["id"], source_url=it.get("guid"),
                        institution=(it.get("dataProvider") or [None])[0], object_number=None, title=title,
                        artist=(it.get("dcCreator") or [None])[0], domain=guess_domain(title, desc, " ".join(it.get("dcType") or [])),
                        culture=normalise_culture(title, desc), culture_raw=None, date_start=_year(year), date_end=_year(year),
                        date_display=year, material=normalise_material(desc), object_type=guess_object_type(title, desc),
                        country=(it.get("country") or [None])[0], provenance=(it.get("provider") or [None])[0],
                        license=normalise_license((it.get("rights") or [None])[0]), license_raw=(it.get("rights") or [None])[0],
                        image_url=(it.get("edmIsShownBy") or [None])[0], external_ids={"europeana_id": it["id"]})
                    if store(c, rec, rec.institution):
                        n += 1
                        print(f"  + {rid} {title} [{rec.license}]")
                    time.sleep(sleep)
                cursor = res.get("nextCursor")
    if n:
        bump_version(f"Europeana: +{n}")
    return n


# ------------------------------------------------------------------ Smithsonian Open Access
def ingest_smithsonian(queries=None, limit=500, sleep=0.2):
    key = os.getenv("SMITHSONIAN_API_KEY")
    if not key:
        print("[SI] set SMITHSONIAN_API_KEY (free at https://api.data.gov/signup/)")
        return 0
    api = "https://api.si.edu/openaccess/api/v1.0/search"
    seen = existing_ids()
    n = 0
    with client() as c:
        for q in (queries or DEFAULT_QUERIES):
            start = 0
            while start < limit:
                res = get_json(c, api, {"api_key": key, "q": f'{q} AND online_media_type:"Images" AND cc0:true', "rows": 100, "start": start})
                rows = ((res or {}).get("response") or {}).get("rows") or []
                if not rows:
                    break
                for it in rows:
                    rid = "SI_" + re.sub(r"[^A-Za-z0-9]+", "_", it["id"])
                    if rid in seen:
                        continue
                    content = it.get("content", {})
                    desc = content.get("descriptiveNonRepeating", {})
                    ff = content.get("freetext", {})
                    idx = content.get("indexedStructured", {})
                    title = it.get("title")
                    media = (desc.get("online_media") or {}).get("media") or []
                    url = next((m.get("content") for m in media if m.get("type") == "Images"), None)
                    if not url or not looks_like_face_object(title, " ".join(idx.get("object_type") or []), " ".join(x.get("content", "") for x in ff.get("notes", []))):
                        continue
                    seen.add(rid)
                    dates = idx.get("date") or []
                    rec = CorpusRecord(
                        id=rid, source="SI", source_object_id=it["id"], source_url=desc.get("record_link"),
                        institution=desc.get("data_source"), object_number=desc.get("record_ID"), title=title,
                        artist=", ".join(idx.get("name") or []) or None, domain=guess_domain(title, " ".join(idx.get("object_type") or [])),
                        culture=normalise_culture(title, " ".join(idx.get("culture") or [])), culture_raw=", ".join(idx.get("culture") or []) or None,
                        date_display=", ".join(dates) or None, date_start=_year(dates[0]) if dates else None, date_end=_year(dates[-1]) if dates else None,
                        material=normalise_material(" ".join(x.get("content", "") for x in ff.get("physicalDescription", []))),
                        object_type=guess_object_type(title, " ".join(idx.get("object_type") or [])), country=", ".join(idx.get("geoLocation", [{}])[0].get("L2", {}).get("content", "") for _ in [0]) or None,
                        license="CC0" if desc.get("metadata_usage", {}).get("access") == "CC0" else "UNKNOWN",
                        license_raw=str(desc.get("metadata_usage")), image_url=url, external_ids={"si_id": it["id"]})
                    if store(c, rec, desc.get("data_source")):
                        n += 1
                        print(f"  + {rid} {title} [{rec.license}]")
                    time.sleep(sleep)
                start += 100
    if n:
        bump_version(f"Smithsonian: +{n}")
    return n


# ------------------------------------------------------------------ Rijksmuseum
def ingest_rijksmuseum(queries=None, limit=500, sleep=0.3):
    """Rijksmuseum Search API (no key) → Linked Art records → IIIF image."""
    api = "https://data.rijksmuseum.nl/search/collection"
    seen = existing_ids()
    n = 0
    with client() as c:
        for q in (queries or DEFAULT_QUERIES):
            res = get_json(c, api, {"title": q, "imageAvailable": "true"})
            ids = [x.get("id") for x in (res or {}).get("orderedItems", []) if x.get("id")][:limit]
            for uri in ids:
                rid = "RIJKS_" + uri.rstrip("/").split("/")[-1]
                if rid in seen:
                    continue
                la = get_json(c, uri)
                time.sleep(sleep)
                if not la:
                    continue
                title = next((x.get("content") for x in la.get("identified_by", []) if x.get("type") == "Name"), None)
                objnum = next((x.get("content") for x in la.get("identified_by", []) if x.get("type") == "Identifier"), None)
                iiif = None
                for rep in la.get("representation", []) or la.get("subject_of", []):
                    for dig in rep.get("digitally_shown_by", []) or rep.get("digitally_carried_by", []):
                        for svc in dig.get("conforms_to", []) or []:
                            pass
                        acc = dig.get("access_point", [{}])[0].get("id")
                        if acc and "iiif" in acc:
                            iiif = acc.split("/full/")[0]
                if not iiif:
                    continue
                prod = (la.get("produced_by") or {})
                ts = (prod.get("timespan") or {})
                ds, de = _year(ts.get("begin_of_the_begin", "")[:4]), _year(ts.get("end_of_the_end", "")[:4])
                cls = " ".join(x.get("_label", "") for x in la.get("classified_as", []))
                if not looks_like_face_object(title, cls):
                    continue
                seen.add(rid)
                rec = CorpusRecord(id=rid, source="RIJKS", source_object_id=uri, source_url=uri, institution="Rijksmuseum", object_number=objnum,
                                   title=title, artist=None, domain=guess_domain(cls, title), culture=normalise_culture(title, cls),
                                   date_start=ds, date_end=de, material=normalise_material(cls), object_type=guess_object_type(title, cls),
                                   license="CC0", license_raw="Rijksmuseum images: CC0 (verify per object)", iiif_url=iiif, external_ids={"linked_art": uri})
                if store(c, rec, "Rijksmuseum"):
                    n += 1
                    print(f"  + {rid} {title}")
    if n:
        bump_version(f"Rijksmuseum: +{n}")
    return n


# ------------------------------------------------------------------ Getty
def ingest_getty(queries=None, limit=300, sleep=0.3):
    """Getty Museum Collection (Linked Art).  Uses the public search endpoint; images via IIIF."""
    api = "https://data.getty.edu/museum/collection/search"
    seen = existing_ids()
    n = 0
    with client() as c:
        for q in (queries or DEFAULT_QUERIES):
            res = get_json(c, api, {"q": q, "type": "HumanMadeObject", "limit": min(limit, 100)})
            items = (res or {}).get("orderedItems") or (res or {}).get("items") or []
            for it in items[:limit]:
                uri = it.get("id")
                if not uri:
                    continue
                rid = "GETTY_" + uri.rstrip("/").split("/")[-1]
                if rid in seen:
                    continue
                la = get_json(c, uri)
                time.sleep(sleep)
                if not la:
                    continue
                title = next((x.get("content") for x in la.get("identified_by", []) if x.get("type") == "Name"), None)
                iiif = None
                for rep in la.get("representation", []):
                    for dig in rep.get("digitally_shown_by", []):
                        acc = (dig.get("access_point") or [{}])[0].get("id")
                        if acc and "iiif" in acc:
                            iiif = acc.split("/full/")[0]
                rights = " ".join(x.get("_label", "") + " " + x.get("id", "") for x in la.get("subject_to", []))
                cls = " ".join(x.get("_label", "") for x in la.get("classified_as", []))
                if not iiif or not looks_like_face_object(title, cls):
                    continue
                seen.add(rid)
                ts = (la.get("produced_by") or {}).get("timespan") or {}
                rec = CorpusRecord(id=rid, source="GETTY", source_object_id=uri, source_url=uri, institution="J. Paul Getty Museum", object_number=None,
                                   title=title, domain=guess_domain(cls, title), culture=normalise_culture(title, cls),
                                   date_start=_year(ts.get("begin_of_the_begin", "")[:4]), date_end=_year(ts.get("end_of_the_end", "")[:4]),
                                   material=normalise_material(cls), object_type=guess_object_type(title, cls),
                                   license=normalise_license(rights), license_raw=rights or None, iiif_url=iiif, external_ids={"linked_art": uri})
                if store(c, rec, "J. Paul Getty Museum"):
                    n += 1
                    print(f"  + {rid} {title} [{rec.license}]")
    if n:
        bump_version(f"Getty: +{n}")
    return n


# ------------------------------------------------------------------ Harvard Art Museums
def ingest_harvard(queries=None, limit=500, sleep=0.2):
    key = os.getenv("HARVARD_API_KEY")
    if not key:
        print("[HARVARD] set HARVARD_API_KEY (free at https://harvardartmuseums.org/collections/api)")
        return 0
    api = "https://api.harvardartmuseums.org/object"
    seen = existing_ids()
    n = 0
    with client() as c:
        for q in (queries or DEFAULT_QUERIES):
            page = 1
            got = 0
            while got < limit:
                res = get_json(c, api, {"apikey": key, "q": q, "hasimage": 1, "size": 100, "page": page})
                recs = (res or {}).get("records") or []
                if not recs:
                    break
                for o in recs:
                    got += 1
                    rid = f"HARVARD_{o['objectid']}"
                    if rid in seen or not o.get("primaryimageurl"):
                        continue
                    if not looks_like_face_object(o.get("title"), o.get("classification"), o.get("medium")):
                        continue
                    seen.add(rid)
                    rec = CorpusRecord(id=rid, source="HARVARD", source_object_id=str(o["objectid"]), source_url=o.get("url"), institution="Harvard Art Museums",
                                       object_number=o.get("objectnumber"), title=o.get("title"), artist=(o.get("people") or [{}])[0].get("name"),
                                       domain=guess_domain(o.get("classification"), o.get("medium")), culture=normalise_culture(o.get("culture"), o.get("period"), o.get("title")),
                                       culture_raw=o.get("culture"), date_start=o.get("datebegin"), date_end=o.get("dateend"), date_display=o.get("dated"),
                                       material=normalise_material(o.get("medium")), material_raw=o.get("medium"), object_type=guess_object_type(o.get("classification"), o.get("title")),
                                       provenance=o.get("creditline"), license=normalise_license(o.get("imagepermissionlevel") == 0 and "public domain" or o.get("copyright")),
                                       license_raw=str(o.get("copyright")), image_url=o.get("primaryimageurl"), iiif_url=(o.get("images") or [{}])[0].get("iiifbaseuri"))
                    if store(c, rec, o.get("creditline")):
                        n += 1
                        print(f"  + {rid} {rec.title} [{rec.license}]")
                    time.sleep(sleep)
                page += 1
    if n:
        bump_version(f"Harvard: +{n}")
    return n


# ------------------------------------------------------------------ Wikimedia Commons (high-recall)
def ingest_wikimedia(categories=None, limit=300, sleep=0.5):
    api = "https://commons.wikimedia.org/w/api.php"
    cats = categories or ["Portrait sculptures", "Ancient Egyptian portrait sculptures", "Roman portrait busts", "Moche portrait vessels",
                          "Maya sculptures", "Benin Bronzes", "Terracotta Army", "Portrait paintings", "Portrait drawings", "Portrait photographs"]
    seen = existing_ids()
    n = 0
    with client() as c:
        for cat in cats:
            cont, got = {}, 0
            while got < limit:
                res = get_json(c, api, {"action": "query", "format": "json", "generator": "categorymembers", "gcmtitle": f"Category:{cat}",
                                        "gcmtype": "file", "gcmlimit": 50, "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1600, **cont})
                if not res or "query" not in res:
                    break
                for page in res["query"]["pages"].values():
                    got += 1
                    info = (page.get("imageinfo") or [{}])[0]
                    em = info.get("extmetadata", {})
                    lic_raw = (em.get("LicenseShortName", {}).get("value") or "") + " " + (em.get("License", {}).get("value") or "")
                    lic = normalise_license(lic_raw)
                    title = page.get("title", "").replace("File:", "")
                    desc = re.sub(r"<[^>]+>", "", em.get("ImageDescription", {}).get("value") or "")
                    if not looks_like_face_object(title, desc, cat):
                        continue
                    rid = f"WIKI_{page['pageid']}"
                    if rid in seen:
                        continue
                    seen.add(rid)
                    rec = CorpusRecord(id=rid, source="WIKI", source_object_id=str(page["pageid"]), source_url=info.get("descriptionurl"),
                                       institution=None, object_number=None, title=title, domain=guess_domain(title, desc, cat),
                                       culture=normalise_culture(title, desc, cat), material=normalise_material(desc), object_type=guess_object_type(title, desc),
                                       provenance=desc[:500] or None, license=lic, license_raw=lic_raw.strip(), rights_holder=re.sub(r"<[^>]+>", "", em.get("Artist", {}).get("value") or "") or None,
                                       image_url=info.get("thumburl") or info.get("url"), external_ids={"commons_pageid": page["pageid"]})
                    if store(c, rec, rec.rights_holder):
                        n += 1
                        print(f"  + {rid} {title[:60]} [{lic}]")
                    time.sleep(sleep)
                if "continue" not in res:
                    break
                cont = res["continue"]
    if n:
        bump_version(f"Wikimedia: +{n}")
    return n


# ------------------------------------------------------------------ Cleveland Museum of Art (CC0, no key)
def ingest_cleveland(queries=None, limit=500, sleep=0.3):
    api = "https://openaccess-api.clevelandart.org/api/artworks/"
    seen = existing_ids()
    n = 0
    with client() as c:
        for q in (queries or DEFAULT_QUERIES):
            res = get_json(c, api, {"q": q, "has_image": 1, "cc0": 1, "limit": min(limit, 1000)})
            for o in (res or {}).get("data", []):
                rid = f"CMA_{o['id']}"
                if rid in seen:
                    continue
                title = o.get("title") or ""
                culture = ", ".join(o.get("culture") or [])
                if not looks_like_face_object(title, o.get("type"), o.get("description")):
                    continue
                url = ((o.get("images") or {}).get("web") or {}).get("url")
                if not url:
                    continue
                seen.add(rid)
                rec = CorpusRecord(id=rid, source="CMA", source_object_id=str(o["id"]), source_url=o.get("url"), institution="Cleveland Museum of Art",
                                   object_number=o.get("accession_number"), title=title, artist=", ".join(x.get("description", "") for x in (o.get("creators") or [])) or None,
                                   domain=guess_domain(o.get("type"), o.get("technique")), culture=normalise_culture(culture, title), culture_raw=culture or None,
                                   date_start=o.get("creation_date_earliest"), date_end=o.get("creation_date_latest"), date_display=o.get("creation_date"),
                                   material=normalise_material(o.get("technique")), material_raw=o.get("technique"), object_type=guess_object_type(o.get("type"), title),
                                   provenance=o.get("creditline"), license="CC0", license_raw="CC0 (Cleveland Museum of Art Open Access)", image_url=url,
                                   external_ids={"cma_id": o["id"]})
                if store(c, rec, o.get("creditline")):
                    n += 1
                    print(f"  + {rid} {title} [CC0]")
                time.sleep(sleep)
    if n:
        bump_version(f"Cleveland: +{n}")
    return n


CONNECTORS = {"cleveland": ingest_cleveland, "met": ingest_met, "artic": ingest_artic, "europeana": ingest_europeana, "smithsonian": ingest_smithsonian,
              "rijksmuseum": ingest_rijksmuseum, "getty": ingest_getty, "harvard": ingest_harvard, "wikimedia": ingest_wikimedia}
