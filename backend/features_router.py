from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from database import SessionLocal, Image as DBImage, Album
import datetime as dt, json, logging
from sqlalchemy import text, func, extract

router = APIRouter()
logger = logging.getLogger("features")


# ── DB migration: add user_tags column if missing ────────────────────────────
def ensure_extra_columns():
    db = SessionLocal()
    try:
        db.execute(text("ALTER TABLE images ADD COLUMN user_tags TEXT DEFAULT '[]'"))
        db.commit()
        logger.info("✅ user_tags column added")
    except Exception:
        db.rollback()   # already exists — fine
    finally:
        db.close()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _live(db):
    return db.query(DBImage).filter(
        (DBImage.is_trashed == False) | (DBImage.is_trashed == None)
    )

def _bare(filename):
    import os
    return os.path.basename(filename) if filename else None

def _row(img):
    return {
        "id":               img.id,
        "filename":         _bare(img.filename),
        "timestamp":        img.timestamp.isoformat() if img.timestamp else None,
        "caption_short":    img.caption_short or "",
        "quality_score":    img.quality_score,
        "quality_level":    img.quality_level,
        "aesthetic_score":  img.aesthetic_score,
        "dominant_emotion": img.dominant_emotion,
        "face_emotion_count": img.face_emotion_count,
        "person_count":     img.person_count or 0,
        "is_favorite":      bool(img.is_favorite),
        "user_tags":        _get_tags_sql(img),
    }

def _get_tags_sql(img):
    try:
        raw = db_execute_scalar(img.id)
        return json.loads(raw) if raw else []
    except Exception:
        return []

def db_execute_scalar(image_id):
    db = SessionLocal()
    try:
        row = db.execute(text("SELECT user_tags FROM images WHERE id=:id"), {"id": image_id}).fetchone()
        return row[0] if row else None
    finally:
        db.close()

def _get_tags(db, image_id):
    try:
        row = db.execute(text("SELECT user_tags FROM images WHERE id=:id"), {"id": image_id}).fetchone()
        return json.loads(row[0]) if row and row[0] else []
    except Exception:
        return []

def _set_tags(db, image_id, tags):
    db.execute(text("UPDATE images SET user_tags=:t WHERE id=:id"),
               {"t": json.dumps(sorted(set(tags))), "id": image_id})


# ════════════════════════════════════════════════════════════════════════════
# EMOTION
# ════════════════════════════════════════════════════════════════════════════
@router.post("/search/emotion")
def search_by_emotion(emotion: str = Form(...), top_k: int = Form(50)):
    db = SessionLocal()
    try:
        imgs = (
            _live(db)
            .filter(DBImage.dominant_emotion == emotion.lower().strip())
            .order_by(DBImage.timestamp.desc())
            .limit(top_k).all()
        )
        if not imgs:
            return {"status": "not_found", "message": f"No photos with emotion '{emotion}'", "results": []}
        return {"status": "found", "emotion": emotion.lower(), "count": len(imgs), "results": [_row(i) for i in imgs]}
    finally:
        db.close()


@router.get("/emotions/summary")
def emotion_summary():
    db = SessionLocal()
    try:
        rows = (
            db.query(DBImage.dominant_emotion, func.count(DBImage.id).label("n"))
            .filter((DBImage.is_trashed == False) | (DBImage.is_trashed == None))
            .filter(DBImage.dominant_emotion != None, DBImage.dominant_emotion != "")
            .group_by(DBImage.dominant_emotion)
            .order_by(func.count(DBImage.id).desc())
            .all()
        )
        return {"emotions": [{"emotion": r[0], "count": r[1]} for r in rows if r[0]]}
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# ON THIS DAY
# ════════════════════════════════════════════════════════════════════════════
@router.get("/on-this-day")
def on_this_day():
    db = SessionLocal()
    try:
        today = dt.date.today()
        imgs = (
            _live(db)
            .filter(
                extract("month", DBImage.timestamp) == today.month,
                extract("day",   DBImage.timestamp) == today.day,
            )
            .order_by(DBImage.timestamp.desc()).all()
        )
        by_year: dict = {}
        for img in imgs:
            if not img.timestamp: continue
            yr = img.timestamp.year
            if yr == today.year: continue
            by_year.setdefault(yr, []).append(_row(img))

        years = sorted(by_year.keys(), reverse=True)
        return {
            "date":  today.strftime("%B %-d"),
            "total": sum(len(v) for v in by_year.values()),
            "years": [{"year": y, "count": len(by_year[y]), "images": by_year[y][:10]} for y in years],
        }
    except Exception as e:
        # %-d not supported on Windows — fallback
        today = dt.date.today()
        return {"date": today.strftime("%B %d"), "total": 0, "years": []}
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# MAP
# ════════════════════════════════════════════════════════════════════════════
@router.get("/map/photos")
def map_photos():
    db = SessionLocal()
    try:
        imgs = (
            _live(db)
            .filter(DBImage.lat != None, DBImage.lon != None)
            .filter(DBImage.lat != 0,    DBImage.lon != 0)
            .all()
        )
        return {
            "count":  len(imgs),
            "photos": [{
                "id":           img.id,
                "filename":     _bare(img.filename),
                "lat":          float(img.lat),
                "lon":          float(img.lon),
                "timestamp":    img.timestamp.isoformat() if img.timestamp else None,
                "caption_short": img.caption_short or "",
                "quality_level": img.quality_level or "",
            } for img in imgs],
        }
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# MANUAL ALBUMS
# ════════════════════════════════════════════════════════════════════════════
@router.post("/albums/create")
def create_album(title: str = Form(...), description: str = Form("")):
    db = SessionLocal()
    try:
        a = Album(title=title.strip(), description=description.strip(), type="manual")
        db.add(a); db.commit(); db.refresh(a)
        return {"status": "success", "id": a.id, "title": a.title}
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@router.post("/albums/{album_id}/add")
def add_to_album(album_id: int, image_ids: str = Form(...)):
    db = SessionLocal()
    try:
        if not db.query(Album).filter(Album.id == album_id).first():
            raise HTTPException(404, "Album not found")
        ids = [int(x) for x in image_ids.split(",") if x.strip().isdigit()]
        db.query(DBImage).filter(DBImage.id.in_(ids)).update(
            {"album_id": album_id}, synchronize_session=False)
        db.commit()
        return {"status": "success", "added": len(ids)}
    except HTTPException: raise
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@router.post("/albums/{album_id}/remove_image")
def remove_from_album(album_id: int, image_id: int = Form(...)):
    db = SessionLocal()
    try:
        db.query(DBImage).filter(DBImage.id == image_id, DBImage.album_id == album_id)\
          .update({"album_id": None}, synchronize_session=False)
        db.commit()
        return {"status": "success"}
    finally:
        db.close()


@router.delete("/albums/{album_id}/delete")
def delete_album(album_id: int):
    db = SessionLocal()
    try:
        a = db.query(Album).filter(Album.id == album_id).first()
        if not a: raise HTTPException(404, "Not found")
        if a.type != "manual": raise HTTPException(400, "Can only delete manual albums")
        db.query(DBImage).filter(DBImage.album_id == album_id)\
          .update({"album_id": None}, synchronize_session=False)
        db.delete(a); db.commit()
        return {"status": "success"}
    except HTTPException: raise
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()

@router.post("/albums/{album_id}/rename")
def rename_album(album_id: int, title: str = Form(...), description: str = Form("")):
    db = SessionLocal()
    try:
        a = db.query(Album).filter(Album.id == album_id).first()
        if not a: raise HTTPException(404, "Album not found")
        a.title = title.strip()
        if description.strip():
            a.description = description.strip()
        db.commit()
        return {"status": "success", "id": a.id, "title": a.title}
    except HTTPException: raise
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


# Event type presets
EVENT_TYPES = [
    "Birthday", "Trip", "Vacation", "Wedding", "Anniversary",
    "Graduation", "Party", "Holiday", "Family", "Work", "Other"
]

@router.get("/event-types")
def get_event_types():
    return {"types": EVENT_TYPES}


@router.post("/events/create")
def create_named_event(
    title: str = Form(...),
    event_type: str = Form("Other"),
    description: str = Form(""),
    date_str: str = Form("")
):
    db = SessionLocal()
    try:
        import re as _re
        clean_title = title.strip()
        if not clean_title:
            raise HTTPException(400, "Title required")
        # Parse optional date
        start_d = None
        if date_str.strip():
            try:
                start_d = dt.datetime.strptime(date_str.strip(), "%Y-%m-%d")
            except Exception:
                pass
        full_desc = f"{event_type}: {description.strip()}" if description.strip() else event_type
        a = Album(
            title=clean_title,
            description=full_desc,
            type="event",
            start_date=start_d,
            end_date=start_d,
        )
        db.add(a); db.commit(); db.refresh(a)
        return {"status": "success", "id": a.id, "title": a.title, "type": a.type}
    except HTTPException: raise
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# USER TAGS
# ════════════════════════════════════════════════════════════════════════════
@router.post("/photo/{image_id}/tags/add")
def add_tag(image_id: int, tag: str = Form(...)):
    db = SessionLocal()
    try:
        if not db.query(DBImage).filter(DBImage.id == image_id).first():
            raise HTTPException(404)
        tags = _get_tags(db, image_id)
        t = tag.strip().lower()
        if t and t not in tags:
            tags.append(t)
            _set_tags(db, image_id, tags)
            db.commit()
        return {"status": "success", "tags": _get_tags(db, image_id)}
    finally:
        db.close()


@router.post("/photo/{image_id}/tags/remove")
def remove_tag(image_id: int, tag: str = Form(...)):
    db = SessionLocal()
    try:
        tags = [x for x in _get_tags(db, image_id) if x != tag.strip().lower()]
        _set_tags(db, image_id, tags)
        db.commit()
        return {"status": "success", "tags": tags}
    finally:
        db.close()


@router.get("/photo/{image_id}/tags")
def get_image_tags(image_id: int):
    db = SessionLocal()
    try:
        return {"tags": _get_tags(db, image_id)}
    finally:
        db.close()


@router.get("/tags")
def all_tags():
    db = SessionLocal()
    try:
        from collections import Counter
        c = Counter()
        for row in db.execute(text("SELECT user_tags FROM images WHERE user_tags IS NOT NULL AND user_tags != '[]'")):
            try:
                for t in json.loads(row[0]): c[t] += 1
            except Exception: pass
        return {"tags": [{"tag": k, "count": v} for k, v in c.most_common()]}
    finally:
        db.close()


@router.get("/tags/{tag}/images")
def images_by_tag(tag: str, top_k: int = Query(100)):
    db = SessionLocal()
    try:
        tl = tag.lower().strip()
        results = []
        # BUGFIX: Use ONLY the exact JSON-array pattern '"%tag%"'.
        # The old pat2 = f"%{tl}%" plain-substring fallback caused false positives:
        # e.g. tag "cat" would match user_tags containing "concatenate".
        pat1 = f'%"{tl}"%'   # stored as JSON array e.g. ["cat","vacation"]
        for row in db.execute(text(
            "SELECT id, filename, timestamp, caption_short, quality_level, quality_score, "
            "dominant_emotion, aesthetic_score, person_count, is_favorite, user_tags "
            "FROM images WHERE user_tags LIKE :pat1 "
            "AND (is_trashed IS NULL OR is_trashed=0)"
        ), {"pat1": pat1}):
            import json as _j
            _tags = []
            try: _tags = _j.loads(row[10] or "[]")
            except Exception: pass
            # BUGFIX: serialize timestamp — raw DB value may be a datetime object
            # or string; frontend expects ISO-format string or null.
            _ts = row[2]
            if _ts is not None:
                _ts = _ts.isoformat() if hasattr(_ts, "isoformat") else str(_ts)
            results.append({
                "id": row[0], "filename": _bare(row[1]),
                "score": 95.0,
                "timestamp": _ts, "caption_short": row[3] or "",
                "quality_level": row[4], "quality_score": row[5] or 0,
                "dominant_emotion": row[6] or "", "aesthetic_score": row[7] or 0,
                "person_count": row[8] or 0, "is_favorite": bool(row[9]),
                "user_tags": _tags, "photo_note": "",
            })
        # Deduplicate by id (shouldn't happen with pat1-only, but keep as safety net)
        seen = set()
        deduped = []
        for r in results:
            if r["id"] not in seen:
                seen.add(r["id"])
                deduped.append(r)
        return {"tag": tl, "count": len(deduped), "results": deduped[:top_k]}
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# BATCH OPERATIONS
# ════════════════════════════════════════════════════════════════════════════
def _parse_ids(raw: str):
    return [int(x) for x in raw.split(",") if x.strip().isdigit()]


@router.post("/batch/favorite")
def batch_favorite(image_ids: str = Form(...), value: int = Form(1)):
    db = SessionLocal()
    try:
        ids = _parse_ids(image_ids)
        db.query(DBImage).filter(DBImage.id.in_(ids)).update(
            {"is_favorite": bool(value)}, synchronize_session=False)
        db.commit()
        return {"status": "success", "updated": len(ids)}
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@router.post("/batch/delete")
def batch_delete(image_ids: str = Form(...)):
    db = SessionLocal()
    try:
        ids = _parse_ids(image_ids)
        db.query(DBImage).filter(DBImage.id.in_(ids)).update(
            {"is_trashed": True, "trashed_at": dt.datetime.now()}, synchronize_session=False)
        db.commit()
        return {"status": "success", "trashed": len(ids)}
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@router.post("/batch/tag")
def batch_tag(image_ids: str = Form(...), tag: str = Form(...)):
    db = SessionLocal()
    try:
        ids = _parse_ids(image_ids); tl = tag.strip().lower(); done = 0
        for img_id in ids:
            tags = _get_tags(db, img_id)
            if tl not in tags:
                tags.append(tl); _set_tags(db, img_id, tags); done += 1
        db.commit()
        return {"status": "success", "tagged": done}
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@router.post("/batch/album")
def batch_album(image_ids: str = Form(...), album_id: int = Form(...)):
    db = SessionLocal()
    try:
        ids = _parse_ids(image_ids)
        db.query(DBImage).filter(DBImage.id.in_(ids)).update(
            {"album_id": album_id}, synchronize_session=False)
        db.commit()
        return {"status": "success", "added": len(ids)}
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# EMOTION TIMELINE
# ════════════════════════════════════════════════════════════════════════════
@router.get("/emotion-timeline")
def emotion_timeline():
    """Return emotion distribution grouped by year-month for charting."""
    db = SessionLocal()
    try:
        from collections import defaultdict
        EMOTIONS = ["happy", "sad", "angry", "neutral", "surprised", "disgusted", "fearful"]
        rows = db.execute(text(
            "SELECT strftime('%Y-%m', timestamp) as ym, dominant_emotion, COUNT(*) as n "
            "FROM images WHERE dominant_emotion IS NOT NULL AND dominant_emotion != '' "
            "AND (is_trashed IS NULL OR is_trashed=0) AND timestamp IS NOT NULL "
            "GROUP BY ym, dominant_emotion ORDER BY ym"
        )).fetchall()
        by_month: dict = defaultdict(lambda: {e: 0 for e in EMOTIONS})
        for ym, emotion, n in rows:
            if ym and emotion:
                by_month[ym][emotion.lower()] = n
        months = sorted(by_month.keys())
        totals = {e: sum(by_month[m].get(e, 0) for m in months) for e in EMOTIONS}
        return {
            "emotions": EMOTIONS,
            "months": months,
            "totals": totals,
            "series": {
                e: [by_month[m].get(e, 0) for m in months]
                for e in EMOTIONS
            },
            "hint": None if any(totals[e]>0 for e in EMOTIONS if e!="neutral") else
                    "Only neutral emotions detected. Install fer (pip install fer) and run Re-index AI."
        }
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# GROUP PHOTO FILTER (person_count >= N)
# ════════════════════════════════════════════════════════════════════════════
@router.get("/group-photos")
def group_photos(min_people: int = Query(2), top_k: int = Query(200)):
    db = SessionLocal()
    try:
        imgs = (
            _live(db)
            .filter(DBImage.person_count >= min_people)
            .order_by(DBImage.timestamp.desc())
            .limit(top_k).all()
        )
        return {"min_people": min_people, "count": len(imgs), "results": [_row(i) for i in imgs]}
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# PERSON CO-OCCURRENCE
# ════════════════════════════════════════════════════════════════════════════
@router.get("/co-occurrence")
def co_occurrence(person_ids: str = Query(...)):
    """Return photos where ALL specified person IDs appear together."""
    db = SessionLocal()
    try:
        from database import Face as DBFace
        ids = [int(x) for x in person_ids.split(",") if x.strip().isdigit()]
        if len(ids) < 2:
            raise HTTPException(400, "Provide at least 2 person_ids")

        # Find image_ids that have ALL requested persons
        from sqlalchemy import and_
        sets = []
        for pid in ids:
            img_ids = {
                row[0]
                for row in db.execute(
                    text("SELECT image_id FROM faces WHERE person_id=:pid"), {"pid": pid}
                ).fetchall()
            }
            sets.append(img_ids)
        common_ids = set.intersection(*sets) if sets else set()

        imgs = (
            _live(db)
            .filter(DBImage.id.in_(common_ids))
            .order_by(DBImage.timestamp.desc())
            .all()
        )
        # Get person names for response
        pnames = {}
        for pid in ids:
            from database import Person as DBPerson
            p = db.query(DBPerson).filter(DBPerson.id == pid).first()
            pnames[pid] = p.name if p else f"Person {pid}"

        return {
            "person_ids": ids,
            "person_names": [pnames[i] for i in ids],
            "count": len(imgs),
            "results": [_row(i) for i in imgs]
        }
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# FACE SIMILARITY SEARCH
# ════════════════════════════════════════════════════════════════════════════
@router.post("/face-similarity")
async def face_similarity(file: UploadFile = File(...), top_k: int = Form(20)):
    """Upload a face crop → find all photos with similar faces via FAISS."""
    import tempfile, os as _os
    db = SessionLocal()
    try:
        from face_engine import face_engine
        # Save upload to temp file
        suffix = _os.path.splitext(file.filename or "face.jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            # detect_faces returns list of {bbox, embedding}
            face_results = face_engine.detect_faces(tmp_path)
        finally:
            _os.unlink(tmp_path)

        if not face_results:
            return {"count": 0, "results": [], "message": "No face detected in uploaded image"}

        # Use first detected face embedding
        query_emb = face_results[0]["embedding"].reshape(1, -1).astype("float32")
        import faiss as _faiss
        _faiss.normalize_L2(query_emb)

        # Search FAISS face index
        if face_engine.face_index is None or face_engine.face_index.ntotal == 0:
            return {"count": 0, "results": [], "message": "Face index is empty — run Re-index AI first"}

        k = min(top_k * 3, face_engine.face_index.ntotal)
        scores, indices = face_engine.face_index.search(query_emb, k)

        # Map FAISS indices → Face DB IDs → Image IDs
        enriched = []
        seen_imgs = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(face_engine.face_id_map):
                continue
            if float(score) < 0.35:   # cosine similarity threshold
                continue
            face_db_id = face_engine.face_id_map[idx]
            face_row = db.query(DBFace).filter(DBFace.id == face_db_id).first()
            if not face_row or not face_row.image_id:
                continue
            img_id = face_row.image_id
            if img_id in seen_imgs:
                continue
            seen_imgs.add(img_id)
            img = db.query(DBImage).filter(DBImage.id == img_id).first()
            if not img or img.is_trashed:
                continue
            enriched.append({**_row(img), "similarity": round(float(score), 3)})
            if len(enriched) >= top_k:
                break

        enriched.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return {"count": len(enriched), "results": enriched}
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# PHOTO NOTES
# ════════════════════════════════════════════════════════════════════════════
@router.post("/photo/{image_id}/note")
def set_note(image_id: int, note: str = Form(...)):
    db = SessionLocal()
    try:
        db.execute(
            text("UPDATE images SET photo_note=:n WHERE id=:id"),
            {"n": note.strip(), "id": image_id}
        )
        db.commit()
        return {"status": "success", "note": note.strip()}
    except Exception as e:
        db.rollback(); raise HTTPException(500, str(e))
    finally:
        db.close()


@router.get("/photo/{image_id}/note")
def get_note(image_id: int):
    db = SessionLocal()
    try:
        row = db.execute(
            text("SELECT photo_note FROM images WHERE id=:id"), {"id": image_id}
        ).fetchone()
        return {"note": row[0] or "" if row else ""}
    finally:
        db.close()


@router.get("/notes/search")
def search_notes(q: str = Query(...), top_k: int = Query(100)):
    """Search photos by their personal notes."""
    db = SessionLocal()
    try:
        term = q.lower().strip()
        rows = db.execute(text(
            "SELECT id, filename, timestamp, caption_short, photo_note, person_count, "
            "quality_level, quality_score, dominant_emotion, aesthetic_score, is_favorite, user_tags "
            "FROM images WHERE photo_note LIKE :pat AND (is_trashed IS NULL OR is_trashed=0) "
            "ORDER BY timestamp DESC LIMIT :lim"
        ), {"pat": f"%{term}%", "lim": top_k}).fetchall()
        results = []
        for r in rows:
            try: ut = json.loads(r[11]) if r[11] else []
            except: ut = []
            results.append({
                "id": r[0], "filename": _bare(r[1]),
                "timestamp": r[2], "caption_short": r[3] or "",
                "photo_note": r[4] or "", "person_count": r[5] or 0,
                "quality_level": r[6], "quality_score": r[7],
                "dominant_emotion": r[8], "aesthetic_score": r[9],
                "is_favorite": bool(r[10]), "user_tags": ut
            })
        return {"query": q, "count": len(results), "results": results}
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# PEOPLE FREQUENCY
# ════════════════════════════════════════════════════════════════════════════
@router.get("/people/frequency")
def people_frequency(top_k: int = Query(20)):
    db = SessionLocal()
    try:
        from database import Person as DBPerson, Face as DBFace
        rows = db.execute(text(
            "SELECT p.id, p.name, COUNT(DISTINCT f.image_id) as n "
            "FROM people p JOIN faces f ON f.person_id=p.id "
            "JOIN images i ON i.id=f.image_id "
            "WHERE (i.is_trashed IS NULL OR i.is_trashed=0) AND i.person_count > 0 "
            "GROUP BY p.id ORDER BY n DESC LIMIT :lim"
        ), {"lim": top_k}).fetchall()
        return {
            "people": [{"id": r[0], "name": r[1] or f"Person {r[0]}", "count": r[2]} for r in rows]
        }
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE SEARCH — 100% OFFLINE via Ollama + rule-based fallback
# ════════════════════════════════════════════════════════════════════════════

def _rule_based_parse(query: str, people_ctx: str) -> dict:
    """
    Offline rule-based NL parser — no model needed.
    Handles: emotions, years, months, group sizes, person names, tags, keywords.
    """
    import re as _re
    q = query.lower().strip()
    plan = {}

    # ── Emotion ──────────────────────────────────────────────────────────
    EMO_MAP = {
        "happy": ["happy", "happiest", "happiness", "smile", "smiling", "joy", "joyful", "laugh", "fun"],
        "sad": ["sad", "sadness", "cry", "crying", "upset", "unhappy", "grief"],
        "angry": ["angry", "anger", "mad", "furious", "rage"],
        "surprised": ["surprised", "surprise", "shocked", "shock", "amazed"],
        "neutral": ["neutral", "calm", "serious", "normal"],
        "disgusted": ["disgusted", "disgust", "gross"],
        "fearful": ["fearful", "fear", "scared", "scared", "anxious"],
    }
    for emo, kws in EMO_MAP.items():
        if any(kw in q for kw in kws):
            plan["emotion"] = emo
            break

    # ── Year ─────────────────────────────────────────────────────────────
    yr = _re.search(r'(19[0-9]{2}|20[0-9]{2})', q)
    if yr:
        plan["year"] = int(yr.group(1))

    # ── Month ────────────────────────────────────────────────────────────
    MONTHS = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
              "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
              "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    for m_name, m_num in MONTHS.items():
        if m_name in q:
            plan["month"] = m_num
            break

    # ── Group size ───────────────────────────────────────────────────────
    GROUP_MAP = [
        (["5+ people","five or more","large group"], 5),
        (["3+ people","three or more","three people","3 people"], 3),
        (["2+ people","two or more","two people","group","together","family","crowd"], 2),
    ]
    for kws, n in GROUP_MAP:
        if any(kw in q for kw in kws):
            plan["min_people"] = n
            break

    # ── Person names (match against known people) ────────────────────────
    if people_ctx and people_ctx != "none":
        ids = []
        for entry in people_ctx.split(","):
            entry = entry.strip()
            if ":" not in entry:
                continue
            pid_str, pname = entry.split(":", 1)
            pname_l = pname.strip().lower()
            if pname_l and pname_l != "unknown":
                parts = pname_l.split()
                if any(part in q for part in parts if len(part) >= 2):
                    try:
                        ids.append(int(pid_str.strip()))
                    except Exception:
                        pass
        if ids:
            plan["person_ids"] = ids

    # ── Tags ─────────────────────────────────────────────────────────────
    tag_match = _re.search(r'tagged?\s+(\w+)', q)
    if tag_match:
        plan["tag"] = tag_match.group(1)

    # ── Note search ───────────────────────────────────────────────────────
    note_match = _re.search(r'(?:note|notes|memo|wrote)\s+(?:about\s+)?(\S.*?)\s*$', q)
    if note_match:
        plan["note_query"] = note_match.group(1).strip()

    # ── Text query (CLIP semantic search) ────────────────────────────────
    # Strip out the structural keywords, use remainder as semantic query
    skip = {"show","find","search","photos","pictures","images","me","my","from","with","in",
            "the","a","an","of","and","or","where","all","get","give","look","for","at",
            str(plan.get("year","")), str(plan.get("month","")),
            "tagged","note","notes","people","group","person","happy","sad","angry",
            "surprised","neutral","disgusted","fearful","happiness","smile","smiling"}
    words = [w for w in _re.sub(r'[^\w\s]','', q).split() if w not in skip and len(w) > 2]
    if words:
        plan["text_query"] = " ".join(words)

    return plan


def _ollama_parse(query: str, people_ctx: str, model: str = "llama3") -> dict:
    """Call local Ollama for NL parsing. Returns parsed plan dict."""
    import urllib.request as _req
    prompt = (
        "You are a photo search assistant. Convert the user's request into a JSON search plan.\n"
        "Available keys (all optional):\n"
        '  "text_query": string for semantic image search\n'
        '  "emotion": one of happy|sad|angry|neutral|surprised|disgusted|fearful\n'
        '  "year": integer\n'
        '  "month": integer 1-12\n'
        '  "min_people": integer\n'
        '  "person_ids": list of integers\n'
        '  "tag": string\n'
        '  "note_query": string\n'
        f"Known people (id:name): {people_ctx}\n"
        "Respond with ONLY a JSON object. No explanation, no markdown.\n\n"
        f"User: {query}"
    )
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 256}
    }).encode()
    req = _req.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with _req.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    raw = body.get("message", {}).get("content", "").strip()
    raw = raw.strip("' \n")
    if raw.startswith("json"): raw = raw[4:].strip()
    return json.loads(raw)


@router.get("/nl-models")
def get_ollama_models():
    """Return list of locally available Ollama models."""
    import urllib.request as _req
    try:
        with _req.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            return {"ollama_available": True, "models": models}
    except Exception:
        return {"ollama_available": False, "models": []}


@router.post("/nl-search")
async def nl_search(query: str = Form(...), model: str = Form("llama3")):
    """
    Parse a natural-language photo query OFFLINE.
    Uses Ollama if running, falls back to built-in rule-based parser automatically.
    No API keys required. 100% offline.
    """
    db = SessionLocal()
    try:
        people_list = db.execute(text("SELECT id, name FROM people WHERE name != 'Unknown' LIMIT 50")).fetchall()
        people_ctx  = ", ".join(f"{r[0]}:{r[1]}" for r in people_list) or "none"

        # Try Ollama first, fall back to rule-based parser
        plan = None
        source = "rule-based"
        try:
            plan = _ollama_parse(query, people_ctx, model)
            source = f"ollama:{model}"
        except Exception as ollama_err:
            logger.info(f"Ollama not available ({ollama_err}), using rule-based parser")
            plan = _rule_based_parse(query, people_ctx)
    except Exception as e:
        raise HTTPException(500, f"Query parsing failed: {e}")
    finally:
        db.close()

    # Execute the plan — call our own endpoints
    results = []
    plan_desc = []

    db2 = SessionLocal()
    try:
        base_q = _live(db2)

        # Apply year/month filter
        if "year" in plan:
            from sqlalchemy import extract
            base_q = base_q.filter(extract("year", DBImage.timestamp) == plan["year"])
            plan_desc.append(f"year {plan['year']}")
        if "month" in plan:
            from sqlalchemy import extract as ext2
            base_q = base_q.filter(ext2("month", DBImage.timestamp) == plan["month"])
            plan_desc.append(f"month {plan['month']}")

        # Emotion filter
        if "emotion" in plan:
            base_q = base_q.filter(DBImage.dominant_emotion == plan["emotion"])
            plan_desc.append(f"emotion={plan['emotion']}")

        # Min people
        if "min_people" in plan:
            base_q = base_q.filter(DBImage.person_count >= plan["min_people"])
            plan_desc.append(f"{plan['min_people']}+ people")

        # Tag filter
        if "tag" in plan:
            tl = plan["tag"].lower().strip()
            base_q = base_q.filter(DBImage.user_tags.contains(f'"{tl}"'))
            plan_desc.append(f"tag={tl}")

        # Note search
        if "note_query" in plan:
            base_q = base_q.filter(DBImage.photo_note.ilike(f"%{plan['note_query']}%"))
            plan_desc.append(f"note~{plan['note_query']}")

        # Person co-occurrence
        if "person_ids" in plan and len(plan["person_ids"]) > 0:
            sets = []
            for pid in plan["person_ids"]:
                img_ids = {
                    row[0] for row in db2.execute(
                        text("SELECT image_id FROM faces WHERE person_id=:pid"), {"pid": pid}
                    ).fetchall()
                }
                sets.append(img_ids)
            common = set.intersection(*sets) if sets else set()
            base_q = base_q.filter(DBImage.id.in_(common))
            plan_desc.append(f"people {plan['person_ids']}")

        imgs = base_q.order_by(DBImage.timestamp.desc()).limit(100).all()

        # If text_query also specified, rank results by CLIP
        if "text_query" in plan and plan["text_query"] and imgs:
            from search_engine import search_engine
            import numpy as np
            emb = search_engine.get_text_embedding(plan["text_query"])
            if emb is not None and search_engine.index is not None:
                import faiss as _faiss
                q = emb.reshape(1, -1).astype("float32")
                _faiss.normalize_L2(q)
                D, I = search_engine.index.search(q, 200)
                scores = {int(I[0][i]): float(D[0][i]) for i in range(len(I[0])) if I[0][i] >= 0}
                imgs.sort(key=lambda x: scores.get(x.id, 0), reverse=True)
            plan_desc.append(f'"{plan["text_query"]}"')

        results = [_row(i) for i in imgs]
    finally:
        db2.close()

    return {
        "plan": plan,
        "description": " + ".join(plan_desc) if plan_desc else "all photos",
        "count": len(results),
        "results": results,
        "source": source
    }