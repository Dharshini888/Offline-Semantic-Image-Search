from fastapi import APIRouter, HTTPException, Query
from database import SessionLocal, Image as DBImage
import json

# Create router for new endpoints
router = APIRouter(prefix="/api/v1", tags=["deep-learning"])


# ============================================================================
# IMAGE DETAILS ENDPOINT
# ============================================================================

@router.get("/image/{image_id}")
def get_image_full_details(image_id: int):
    """
    Get complete image metadata including all new DL features
    
    Example: GET /api/v1/image/42
    
    Response includes:
    - Basic metadata (filename, date, size)
    - Quality metrics (sharpness, exposure, contrast, composition)
    - Generated captions (short and detailed)
    - Extracted text and keywords
    - Emotion detection results
    - Aesthetic scoring
    """
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return {
            "id": img.id,
            "filename": img.filename,
            "timestamp": img.timestamp.isoformat() if img.timestamp else None,
            "size": img.size_bytes,
            "dimensions": {"width": img.width, "height": img.height},
            
            # Quality metrics
            "quality": {
                "overall_score": img.quality_score,
                "level": img.quality_level,
                "metrics": {
                    "sharpness": img.sharpness,
                    "exposure": img.exposure,
                    "contrast": img.contrast,
                    "composition": img.composition
                }
            },
            
            # Image captions
            "captions": {
                "short": img.caption_short,
                "detailed": img.caption_detailed,
                "visual_qa": json.loads(img.caption_vqa or "{}")
            },
            
            # Text extraction
            "text": {
                "ocr_enhanced": img.ocr_text_enhanced,
                "keywords": json.loads(img.ocr_keywords or "[]"),
                "ocr_confidence": img.ocr_confidence,
                "detected_language": img.detected_language
            },
            
            # Emotion detection
            "emotions": {
                "dominant": img.dominant_emotion,
                "face_count": img.face_emotion_count or 0,
                "all_emotions": json.loads(img.emotion_data or "[]")
            },
            
            # Aesthetics
            "aesthetics": {
                "score": img.aesthetic_score,
                "rating": img.aesthetic_rating
            },
            
            # Scene and objects
            "scene": img.scene_label,
            "person_count": img.person_count
        }
    
    finally:
        db.close()


# ============================================================================
# QUALITY ENDPOINTS
# ============================================================================

@router.get("/image/{image_id}/quality")
def get_image_quality(image_id: int):
    """Get quality metrics for an image"""
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return {
            "overall_score": img.quality_score or 0,
            "level": img.quality_level or "Unknown",
            "sharpness": img.sharpness or 0,
            "exposure": img.exposure or 0,
            "contrast": img.contrast or 0,
            "composition": img.composition or 0
        }
    
    finally:
        db.close()


@router.get("/image/{image_id}/caption")
def get_image_caption(image_id: int):
    """Get generated captions for an image"""
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return {
            "short": img.caption_short or "",
            "detailed": img.caption_detailed or "",
            "visual_qa": json.loads(img.caption_vqa or "{}")
        }
    
    finally:
        db.close()


@router.get("/image/{image_id}/text")
def get_image_text(image_id: int):
    """Get OCR extracted text from an image"""
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return {
            "full_text": img.ocr_text_enhanced or "",
            "keywords": json.loads(img.ocr_keywords or "[]"),
            "confidence": img.ocr_confidence or 0.0,
            "language": img.detected_language or "unknown"
        }
    
    finally:
        db.close()


@router.get("/image/{image_id}/emotions")
def get_image_emotions(image_id: int):
    """Get emotion detection results"""
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        
        emotions = json.loads(img.emotion_data or "[]")
        
        return {
            "dominant_emotion": img.dominant_emotion or "neutral",
            "face_count": img.face_emotion_count or 0,
            "emotions": emotions
        }
    
    finally:
        db.close()


@router.get("/image/{image_id}/aesthetics")
def get_image_aesthetics(image_id: int):
    """Get aesthetic scoring"""
    db = SessionLocal()
    try:
        img = db.query(DBImage).filter(DBImage.id == image_id).first()
        
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return {
            "score": img.aesthetic_score or 0,
            "rating": img.aesthetic_rating or "Unknown"
        }
    
    finally:
        db.close()


# ============================================================================
# STATISTICS ENDPOINTS
# ============================================================================

@router.get("/stats/quality")
def get_quality_statistics():
    """Get statistics on image quality across library"""
    db = SessionLocal()
    try:
        import numpy as np
        
        images = db.query(DBImage).all()
        
        if not images:
            return {"message": "No images in database"}
        
        # Calculate statistics
        quality_scores = [img.quality_score for img in images if img.quality_score]
        sharpness_scores = [img.sharpness for img in images if img.sharpness]
        exposure_scores = [img.exposure for img in images if img.exposure]
        aesthetic_scores = [img.aesthetic_score for img in images if img.aesthetic_score]
        
        quality_levels = {}
        for img in images:
            if img.quality_level:
                quality_levels[img.quality_level] = quality_levels.get(img.quality_level, 0) + 1
        
        emotions_count = len([img for img in images if img.dominant_emotion])
        captions_count = len([img for img in images if img.caption_short])
        
        return {
            "total_images": len(images),
            "quality": {
                "average": float(np.mean(quality_scores)) if quality_scores else 0,
                "min": float(np.min(quality_scores)) if quality_scores else 0,
                "max": float(np.max(quality_scores)) if quality_scores else 0,
                "by_level": quality_levels
            },
            "sharpness": {
                "average": float(np.mean(sharpness_scores)) if sharpness_scores else 0
            },
            "exposure": {
                "average": float(np.mean(exposure_scores)) if exposure_scores else 0
            },
            "aesthetics": {
                "average": float(np.mean(aesthetic_scores)) if aesthetic_scores else 0
            },
            "emotions_detected": emotions_count,
            "captions_generated": captions_count
        }
    
    finally:
        db.close()


@router.get("/stats/emotions")
def get_emotion_statistics():
    """Get statistics on emotions detected"""
    db = SessionLocal()
    try:
        images = db.query(DBImage).all()
        
        emotion_counts = {}
        total_faces_with_emotions = 0
        
        for img in images:
            if img.dominant_emotion:
                emotion_counts[img.dominant_emotion] = emotion_counts.get(img.dominant_emotion, 0) + 1
                total_faces_with_emotions += img.face_emotion_count or 0
        
        return {
            "total_emotions_detected": total_faces_with_emotions,
            "breakdown": emotion_counts
        }
    
    finally:
        db.close()


@router.get("/stats/text")
def get_text_statistics():
    """Get statistics on OCR and text extraction"""
    db = SessionLocal()
    try:
        images = db.query(DBImage).all()
        
        with_ocr = len([img for img in images if img.ocr_text_enhanced])
        with_keywords = len([img for img in images if img.ocr_keywords])
        
        avg_confidence = 0
        confidences = [img.ocr_confidence for img in images if img.ocr_confidence]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        
        return {
            "images_with_text": with_ocr,
            "images_with_keywords": with_keywords,
            "average_ocr_confidence": avg_confidence
        }
    
    finally:
        db.close()


# ============================================================================
# ADVANCED SEARCH ENDPOINT
# ============================================================================

@router.get("/search/advanced")
def advanced_search(
    query: str = Query(..., description="Search query"),
    min_quality: float = Query(0, ge=0, le=100, description="Minimum quality score"),
    emotion: str = Query(None, description="Filter by emotion (happy, sad, angry, etc)"),
    min_aesthetic: float = Query(0, ge=0, le=10, description="Minimum aesthetic score"),
    has_text: bool = Query(None, description="Only images with extracted text"),
    top_k: int = Query(20, ge=1, le=100, description="Number of results")
):
    """
    Advanced search with deep learning filters
    
    Examples:
    - /search/advanced?query=dog&min_quality=75
    - /search/advanced?query=sunset&emotion=happy
    - /search/advanced?query=text&has_text=true
    - /search/advanced?query=portrait&min_aesthetic=7&min_quality=80
    """
    db = SessionLocal()
    try:
        # Get all images as base set
        images = db.query(DBImage).all()
        
        results = []
        
        for img in images:
            # Apply quality filter
            if img.quality_score and img.quality_score < min_quality:
                continue
            
            # Apply emotion filter
            if emotion and (not img.dominant_emotion or img.dominant_emotion.lower() != emotion.lower()):
                continue
            
            # Apply aesthetic filter
            if min_aesthetic > 0 and img.aesthetic_score and img.aesthetic_score < min_aesthetic:
                continue
            
            # Apply text filter
            if has_text is True and not img.ocr_text_enhanced:
                continue
            
            # Add to results
            results.append({
                "image_id": img.id,
                "filename": img.filename,
                "caption": img.caption_short or "No caption",
                "quality_score": img.quality_score or 0,
                "emotion": img.dominant_emotion or "neutral",
                "aesthetic_score": img.aesthetic_score or 0,
                "has_text": bool(img.ocr_text_enhanced)
            })
        
        # Return top K results
        return {
            "query": query,
            "filters": {
                "min_quality": min_quality,
                "emotion": emotion,
                "min_aesthetic": min_aesthetic,
                "has_text": has_text
            },
            "total_results": len(results),
            "results": results[:top_k]
        }
    
    finally:
        db.close()


# ============================================================================
# FILTER & BROWSE ENDPOINTS
# ============================================================================

@router.get("/browse/by-quality")
def browse_by_quality(quality_level: str = Query("Good")):
    """
    Browse images by quality level
    
    Levels: Excellent, Good, Fair, Poor
    """
    db = SessionLocal()
    try:
        images = db.query(DBImage).filter(
            DBImage.quality_level == quality_level
        ).all()
        
        return {
            "quality_level": quality_level,
            "count": len(images),
            "images": [
                {
                    "id": img.id,
                    "filename": img.filename,
                    "quality_score": img.quality_score
                }
                for img in images[:50]  # Limit to 50
            ]
        }
    
    finally:
        db.close()


@router.get("/browse/by-emotion")
def browse_by_emotion(emotion: str = Query("happy")):
    """
    Browse images by detected emotion
    
    Emotions: happy, sad, angry, neutral, surprised, disgusted, fearful
    """
    db = SessionLocal()
    try:
        images = db.query(DBImage).filter(
            DBImage.dominant_emotion == emotion.lower()
        ).all()
        
        return {
            "emotion": emotion,
            "count": len(images),
            "images": [
                {
                    "id": img.id,
                    "filename": img.filename,
                    "faces_with_emotion": img.face_emotion_count
                }
                for img in images[:50]
            ]
        }
    
    finally:
        db.close()


@router.get("/browse/by-aesthetic")
def browse_by_aesthetic(min_score: float = Query(8.0, ge=0, le=10)):
    """
    Browse best-composed images by aesthetic score
    """
    db = SessionLocal()
    try:
        images = db.query(DBImage).filter(
            DBImage.aesthetic_score >= min_score
        ).order_by(DBImage.aesthetic_score.desc()).all()
        
        return {
            "min_aesthetic_score": min_score,
            "count": len(images),
            "images": [
                {
                    "id": img.id,
                    "filename": img.filename,
                    "aesthetic_score": img.aesthetic_score,
                    "rating": img.aesthetic_rating
                }
                for img in images[:50]
            ]
        }
    
    finally:
        db.close()


@router.get("/browse/with-captions")
def browse_images_with_captions(limit: int = Query(20, ge=1, le=100)):
    """
    Browse images that have auto-generated captions
    """
    db = SessionLocal()
    try:
        images = db.query(DBImage).filter(
            DBImage.caption_short != None
        ).limit(limit).all()
        
        return {
            "count": len(images),
            "images": [
                {
                    "id": img.id,
                    "filename": img.filename,
                    "caption": img.caption_short
                }
                for img in images
            ]
        }
    
    finally:
        db.close()


@router.get("/browse/with-text")
def browse_images_with_extracted_text(limit: int = Query(20, ge=1, le=100)):
    """
    Browse images that have extracted text
    """
    db = SessionLocal()
    try:
        images = db.query(DBImage).filter(
            DBImage.ocr_text_enhanced != None
        ).limit(limit).all()
        
        return {
            "count": len(images),
            "images": [
                {
                    "id": img.id,
                    "filename": img.filename,
                    "text_preview": (img.ocr_text_enhanced[:100] + "...") if img.ocr_text_enhanced else "",
                    "confidence": img.ocr_confidence
                }
                for img in images
            ]
        }
    
    finally:
        db.close()


# ============================================================================
# RECOMMENDATIONS ENDPOINT
# ============================================================================

@router.get("/image/{image_id}/similar")
def get_similar_images(image_id: int, limit: int = Query(10, ge=1, le=50)):
    """
    Get similar images based on:
    - Scene/objects (scene_label)
    - Quality level
    - Aesthetic score
    - Emotions (if applicable)
    """
    db = SessionLocal()
    try:
        target = db.query(DBImage).filter(DBImage.id == image_id).first()
        
        if not target:
            raise HTTPException(status_code=404, detail="Image not found")
        
        all_images = db.query(DBImage).filter(DBImage.id != image_id).all()
        
        # Score similarity
        scored = []
        for img in all_images:
            score = 0.0
            
            # Scene similarity
            if target.scene_label and img.scene_label:
                target_objects = set(target.scene_label.lower().split(","))
                img_objects = set(img.scene_label.lower().split(","))
                overlap = len(target_objects & img_objects)
                score += (overlap / max(len(target_objects), len(img_objects))) * 40
            
            # Quality level similarity
            if target.quality_level and img.quality_level:
                if target.quality_level == img.quality_level:
                    score += 30
            
            # Aesthetic similarity
            if target.aesthetic_score and img.aesthetic_score:
                diff = abs(target.aesthetic_score - img.aesthetic_score)
                score += (10 - min(diff, 10)) * 2
            
            # Emotion similarity
            if target.dominant_emotion and img.dominant_emotion:
                if target.dominant_emotion == img.dominant_emotion:
                    score += 20
            
            scored.append((img, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "target_image": target.filename,
            "similar_count": len(scored),
            "similar_images": [
                {
                    "id": img.id,
                    "filename": img.filename,
                    "similarity_score": s,
                    "scene": img.scene_label,
                    "quality": img.quality_level
                }
                for img, s in scored[:limit]
            ]
        }
    
    finally:
        db.close()


# ============================================================================
# EXPORT FOR USE IN MAIN.PY
# ============================================================================
# In your main.py, add this to import and use:
# from api_endpoints import router
# app.include_router(router)
