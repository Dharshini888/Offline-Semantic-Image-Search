from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import os
import logging
from sqlalchemy import text

Base = declarative_base()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "data", "db.sqlite"))
DB_URL = f"sqlite:///{DB_PATH}"

logger = logging.getLogger("database")


class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, nullable=False)
    original_path = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Existing metadata
    make = Column(String)
    model = Column(String)
    lat = Column(Float)
    lon = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    size_bytes = Column(Integer)
    avg_r = Column(Float, default=0.0)
    avg_g = Column(Float, default=0.0)
    avg_b = Column(Float, default=0.0)
    
    # Content info
    ocr_text = Column(Text)
    scene_label = Column(String)
    is_duplicate = Column(Boolean, default=False)
    duplicate_of = Column(Integer, ForeignKey('images.id'))
    person_count = Column(Integer, default=0)
    is_favorite = Column(Boolean, default=False)
    is_trashed = Column(Boolean, default=False)
    trashed_at = Column(DateTime, nullable=True)
    
    # NEW: Enhanced OCR (EasyOCR)
    ocr_text_enhanced = Column(Text)
    ocr_keywords = Column(Text)
    ocr_confidence = Column(Float)
    detected_language = Column(String)
    
    # NEW: Image Captioning (BLIP)
    caption_short = Column(String)
    caption_detailed = Column(Text)
    caption_vqa = Column(Text)
    caption_timestamp = Column(DateTime)
    
    # NEW: Quality Assessment
    quality_score = Column(Float)
    quality_level = Column(String)
    sharpness = Column(Float)
    exposure = Column(Float)
    contrast = Column(Float)
    composition = Column(Float)
    
    # NEW: Emotion Detection
    emotion_data = Column(Text)
    dominant_emotion = Column(String)
    face_emotion_count = Column(Integer)
    
    # NEW: Aesthetic Scoring
    aesthetic_score = Column(Float)
    aesthetic_rating = Column(String)
    
    # NEW: User-defined tags (JSON array stored as text)
    user_tags = Column(Text, default='[]')

    # NEW: Personal photo note
    photo_note = Column(Text, default='')

    # Relationships
    faces = relationship("Face", back_populates="image")
    album_id = Column(Integer, ForeignKey('albums.id'))
    album = relationship("Album", back_populates="images")


class Face(Base):
    __tablename__ = 'faces'
    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey('images.id'))
    bbox = Column(String)
    embedding_idx = Column(Integer)
    face_embedding = Column(LargeBinary)
    
    person_id = Column(Integer, ForeignKey('people.id'))
    image = relationship("Image", back_populates="faces")
    person = relationship("Person", back_populates="faces")


class Person(Base):
    __tablename__ = 'people'
    id = Column(Integer, primary_key=True)
    name = Column(String, default="Unknown")
    cover_face_id = Column(Integer)
    faces = relationship("Face", back_populates="person")


class Album(Base):
    __tablename__ = 'albums'
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    type = Column(String)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    cover_image_id = Column(Integer)
    images = relationship("Image", back_populates="album")


# Database setup
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database with all tables"""
    if not os.path.exists("../data"):
        os.makedirs("../data")
    
    Base.metadata.create_all(bind=engine)
    
    # Run migrations for new columns
    with engine.begin() as conn:
        try:
            res = conn.execute(text("PRAGMA table_info(images)"))
            existing_cols = {row[1] for row in res.fetchall()}
            
            new_columns = {
                "ocr_text_enhanced": "TEXT",
                "ocr_keywords": "TEXT",
                "ocr_confidence": "FLOAT",
                "detected_language": "STRING",
                "caption_short": "STRING",
                "caption_detailed": "TEXT",
                "caption_vqa": "TEXT",
                "caption_timestamp": "DATETIME",
                "quality_score": "FLOAT",
                "quality_level": "STRING",
                "sharpness": "FLOAT",
                "exposure": "FLOAT",
                "contrast": "FLOAT",
                "composition": "FLOAT",
                "emotion_data": "TEXT",
                "dominant_emotion": "STRING",
                "face_emotion_count": "INTEGER",
                "aesthetic_score": "FLOAT",
                "aesthetic_rating": "STRING",
                "user_tags": "TEXT DEFAULT '[]'",
                "photo_note": "TEXT DEFAULT ''",
            }
            
            for col_name, col_type in new_columns.items():
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE images ADD COLUMN {col_name} {col_type}"))
                    logger.info(f"Added column: {col_name}")
        
        except Exception as e:
            logger.error(f"Migration error: {e}")


if __name__ == "__main__":
    init_db()
    print("Database initialized.")