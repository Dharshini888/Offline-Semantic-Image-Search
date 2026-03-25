import os
import json
import logging
import datetime
from tqdm import tqdm
import exifread
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed

from database import init_db, SessionLocal, Image as DBImage, Face as DBFace, Person, Album
from search_engine import search_engine
from face_engine import face_engine
from ocr_engine import extract_text
from duplicate_engine import duplicate_engine
from clustering_engine import clustering_engine
from detector_engine import detector_engine

# Paths
IMAGE_DIR = "../data/images"
FAISS_INDEX_PATH = "../data/index.faiss"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BuildIndex")

def extract_exif(path):
    """Extract EXIF metadata including GPS coordinates"""
    data = {"lat": None, "lon": None, "date": None, "make": None, "model": None}
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            # Date
            if 'EXIF DateTimeOriginal' in tags:
                date_str = str(tags['EXIF DateTimeOriginal'])
                try:
                    data['date'] = datetime.datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                except: 
                    pass
            
            # Make/Model
            data['make'] = str(tags.get('Image Make', ''))
            data['model'] = str(tags.get('Image Model', ''))
            
            # GPS
            try:
                if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                    lat_data = tags['GPS GPSLatitude'].values
                    lon_data = tags['GPS GPSLongitude'].values
                    
                    lat = float(lat_data[0]) + float(lat_data[1])/60.0 + float(lat_data[2])/3600.0
                    lon = float(lon_data[0]) + float(lon_data[1])/60.0 + float(lon_data[2])/3600.0
                    
                    if 'GPS GPSLatitudeRef' in tags and str(tags['GPS GPSLatitudeRef']) == 'S':
                        lat = -lat
                    if 'GPS GPSLongitudeRef' in tags and str(tags['GPS GPSLongitudeRef']) == 'W':
                        lon = -lon
                    
                    data['lat'] = lat
                    data['lon'] = lon
            except Exception as e:
                logger.debug(f"GPS extraction failed: {e}")
    
    except Exception as e:
        logger.debug(f"EXIF extraction failed: {e}")
    
    return data

def extract_average_color(image_path):
    """
    Extract average RGB color from the center of the image.
    Avoids borders which often have irrelevant colors.
    """
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return 0.0, 0.0, 0.0
        
        # Sample center region (avoid borders)
        h, w = img.shape[:2]
        center = img[h//4:3*h//4, w//4:3*w//4]
        
        # BGR order in OpenCV
        avg_b = float(np.mean(center[:,:,0]))
        avg_g = float(np.mean(center[:,:,1]))
        avg_r = float(np.mean(center[:,:,2]))
        
        return avg_r, avg_g, avg_b
    except Exception as e:
        logger.warning(f"Color extraction failed for {image_path}: {e}")
        return 0.0, 0.0, 0.0

def process_image(filename, IMAGE_DIR):
    """
    Process a single image: extract embeddings, faces, OCR, color, etc.
    This function is called in parallel by ThreadPoolExecutor.
    """
    path = os.path.join(IMAGE_DIR, filename)
    
    try:
        # 1. Semantic Embedding (CLIP)
        clip_emb = search_engine.get_image_embedding(path)
        if clip_emb is None:
            logger.warning(f"⚠️  CLIP embedding failed for {filename}")
            return None
        
        # 2. OCR Text
        ocr_text = extract_text(path)
        
        # 3. EXIF Data
        exif = extract_exif(path)
        
        # 4. Image dimensions
        from PIL import Image as PILImage
        try:
            img_pil = PILImage.open(path)
            width, height = img_pil.size
        except:
            width, height = None, None
        
        # 5. File size
        file_size = os.path.getsize(path)
        
        # 6. Average color (NEW!)
        avg_r, avg_g, avg_b = extract_average_color(path)
        
        # 7. Person count
        person_count = detector_engine.detect_persons(path)
        
        # 8. Object tags (NEW!)
        object_tags = detector_engine.detect_objects(path, threshold=0.5)
        scene_label = ", ".join(object_tags) if object_tags else None
        
        # 9. Face Detection
        faces = face_engine.detect_faces(path)
        
        return {
            'filename': filename,
            'path': path,
            'clip_emb': clip_emb,
            'ocr_text': ocr_text,
            'exif': exif,
            'width': width,
            'height': height,
            'file_size': file_size,
            'avg_r': avg_r,
            'avg_g': avg_g,
            'avg_b': avg_b,
            'person_count': person_count,
            'scene_label': scene_label,
            'faces': faces,
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to process {filename}: {e}")
        return None

def build_index(num_workers=4):
    """
    Build index with parallel processing.
    
    Args:
        num_workers: Number of parallel workers (default 4)
        
    Time complexity:
    - Sequential: 18 images * 1.2s = 21.6s
    - Parallel (4 workers): 18/4 * 1.2s = 5.4s
    - Speedup: ~4x
    """
    
    init_db()
    db = SessionLocal()
    
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        logger.warning(f"Created empty image directory at {IMAGE_DIR}. Add images and run again.")
        return

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        logger.warning("No images found to process")
        return
    
    logger.info(f"📊 Processing {len(image_files)} images with {num_workers} parallel workers...")

    all_clip_embeddings = []
    all_clip_ids = []
    all_face_embeddings = []
    all_face_db_ids = []
    
    # STEP 1: Parallel image processing
    logger.info(f"⚡ Processing images in parallel ({num_workers} workers)...")
    
    processed_images = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_image, f, IMAGE_DIR): f for f in image_files}
        
        # Process results as they complete
        with tqdm(total=len(image_files), desc="Images processed") as pbar:
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_images.append(result)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Exception processing {filename}: {e}")
                    pbar.update(1)
    
    logger.info(f"✅ Processed {len(processed_images)} images successfully")
    
    # STEP 2: Save to database and build indexes
    logger.info(f"💾 Saving to database...")
    
    for result in processed_images:
        # Check if already exists
        existing = db.query(DBImage).filter(DBImage.filename == result['filename']).first()
        if existing:
            logger.debug(f"Skipping {result['filename']} (already in DB)")
            continue
        
        # Create image record
        img_record = DBImage(
            filename=result['filename'],
            original_path=result['path'],
            timestamp=result['exif']['date'] or datetime.datetime.fromtimestamp(os.path.getmtime(result['path'])),
            make=result['exif']['make'],
            model=result['exif']['model'],
            lat=result['exif']['lat'],
            lon=result['exif']['lon'],
            width=result['width'],
            height=result['height'],
            size_bytes=result['file_size'],
            avg_r=result['avg_r'],
            avg_g=result['avg_g'],
            avg_b=result['avg_b'],
            ocr_text=result['ocr_text'],
            scene_label=result['scene_label'],
            person_count=result['person_count'],
        )
        db.add(img_record)
        db.flush()
        
        # Store CLIP embedding
        all_clip_embeddings.append(result['clip_emb'])
        all_clip_ids.append(img_record.id)
        
        # Process faces
        try:
            for face in result['faces']:
                emb = face['embedding'].astype(np.float32)
                emb_blob = emb.tobytes()
                
                face_record = DBFace(
                    image_id=img_record.id,
                    bbox=json.dumps(face['bbox']),
                    face_embedding=emb_blob,
                )
                db.add(face_record)
                db.flush()
                
                all_face_embeddings.append(emb)
                all_face_db_ids.append(face_record.id)
        except Exception as e:
            logger.warning(f"Face processing failed for {result['filename']}: {e}")
    
    db.commit()
    logger.info(f"✅ Saved {len(all_clip_embeddings)} images to database")
    
    # STEP 3: Build Image FAISS Index
    if all_clip_embeddings:
        logger.info("🔨 Building image FAISS index...")
        clip_data = np.array(all_clip_embeddings).astype('float32')
        faiss.normalize_L2(clip_data)
        
        sub_index = faiss.IndexFlatIP(clip_data.shape[1])
        index = faiss.IndexIDMap(sub_index)
        
        ids_np = np.array(all_clip_ids).astype('int64')
        index.add_with_ids(clip_data, ids_np)
        
        faiss.write_index(index, FAISS_INDEX_PATH)
        logger.info(f"✅ Built image FAISS index with {len(all_clip_embeddings)} vectors")
    
    # STEP 4: Build Face FAISS Index
    if all_face_embeddings:
        logger.info("🔨 Building face FAISS index...")
        face_engine.rebuild_index(all_face_embeddings, all_face_db_ids)
        logger.info(f"✅ Built face FAISS index with {len(all_face_embeddings)} vectors")
    
    # STEP 5: Face Clustering
    logger.info("👥 Clustering faces...")
    if all_face_embeddings:
        labels = face_engine.cluster_faces(all_face_embeddings)

        # CRITICAL: use all_face_db_ids (built in sync with all_face_embeddings)
        # NOT a fresh db.query — the DB query returns faces in an arbitrary order
        # that does NOT match the order of all_face_embeddings, so
        # face_records[i].person_id would assign the wrong person to the wrong face.
        person_count = 0
        person_map = {}

        for i, label in enumerate(labels):
            if label == -1:
                continue

            if label not in person_map:
                new_person = Person(name=f"Person {label + 1}")
                db.add(new_person)
                db.flush()
                person_map[label] = new_person.id
                person_count += 1

            face_db_id = all_face_db_ids[i]
            db.query(DBFace).filter(DBFace.id == face_db_id).update(
                {"person_id": person_map[label]}
            )

        db.commit()
        logger.info(f"✅ Clustered faces into {person_count} people")
    
    # STEP 6: Album Clustering
    logger.info("📅 Detecting events...")
    all_images = db.query(DBImage).all()
    
    if all_images:
        metadata = [
            {
                "id": img.id,
                "lat": img.lat or 0.0,
                "lon": img.lon or 0.0,
                "timestamp": img.timestamp
            }
            for img in all_images if img.timestamp
        ]
        
        if metadata:
            album_labels = clustering_engine.detect_events(metadata)
            album_map = {}
            album_count = 0
            
            for i, label in enumerate(album_labels):
                if label == -1:
                    continue
                
                if label not in album_map:
                    cluster_imgs = [metadata[j] for j, l in enumerate(album_labels) if l == label]
                    ts_list = [m['timestamp'] for m in cluster_imgs if m['timestamp']]
                    
                    start_d = min(ts_list) if ts_list else None
                    end_d = max(ts_list) if ts_list else None
                    
                    # Readable title: "Jan 5, 2024" or "Jan 5 – Jan 7, 2024"
                    if start_d:
                        def _fmt(dt):
                            return dt.strftime("%b ") + str(dt.day) + dt.strftime(", %Y")
                        if end_d and end_d.date() != start_d.date():
                            title = f"{start_d.strftime('%b ') + str(start_d.day)} – {_fmt(end_d)}"
                        else:
                            title = _fmt(start_d)
                    else:
                        title = f"Event {label + 1}"
                    
                    new_album = Album(
                        title=title,
                        type="event",
                        start_date=start_d,
                        end_date=end_d
                    )
                    db.add(new_album)
                    db.flush()
                    album_map[label] = new_album.id
                    album_count += 1
                
                # Fix: use metadata[i]["id"] to update the correct DB row.
                # OLD BUG: all_images[i].album_id = album_map[label]
                # all_images includes ALL images (some have no timestamp and
                # were excluded from metadata), so all_images[i] is a
                # different image than metadata[i] → albums assigned to
                # completely wrong photos.
                img_id = metadata[i]["id"]
                db.query(DBImage).filter(DBImage.id == img_id).update(
                    {"album_id": album_map[label]}
                )
            
            db.commit()
            logger.info(f"✅ Created {album_count} albums/events")
    
    logger.info("✅ Build complete!")
    db.close()

if __name__ == "__main__":
    build_index(num_workers=4)
    