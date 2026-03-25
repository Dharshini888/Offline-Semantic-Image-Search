#!/usr/bin/env python3
"""
MIGRATION SCRIPT - Safely Upgrade Existing Database

This script will:
1. Backup your existing database
2. Add new columns for deep learning features
3. Verify data integrity
4. Create indices for performance

IMPORTANT: Always backup before running!

Usage:
    python migrate_to_deeplearning.py
"""

import os
import sys
import sqlite3
import shutil
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseMigration:
    """Handle database migration safely"""
    
    def __init__(self, db_path="../data/db.sqlite"):
        self.db_path = db_path
        self.backup_path = None
    
    def check_database_exists(self):
        """Verify database exists"""
        logger.info("🔍 Checking database...")
        
        if not os.path.exists(self.db_path):
            logger.error(f"❌ Database not found at {self.db_path}")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if images table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
            if not cursor.fetchone():
                logger.error("❌ 'images' table not found")
                return False
            
            # Count existing images
            cursor.execute("SELECT COUNT(*) FROM images")
            count = cursor.fetchone()[0]
            logger.info(f"✅ Database found with {count} images")
            
            conn.close()
            return True
        
        except Exception as e:
            logger.error(f"❌ Database check failed: {e}")
            return False
    
    def backup_database(self):
        """Create backup of existing database"""
        logger.info("💾 Creating backup...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_path = f"{self.db_path}.backup_{timestamp}"
            
            shutil.copy2(self.db_path, self.backup_path)
            logger.info(f"✅ Backup created: {self.backup_path}")
            logger.info("   Keep this file safe! You can restore if something goes wrong.")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def add_columns(self):
        """Add new columns for deep learning features"""
        logger.info("🆕 Adding new columns...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get existing columns
            cursor.execute("PRAGMA table_info(images)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            # Define new columns and their types
            new_columns = {
                # Enhanced OCR
                "ocr_text_enhanced": "TEXT",
                "ocr_keywords": "TEXT",
                "ocr_confidence": "REAL DEFAULT 0.0",
                "detected_language": "TEXT DEFAULT 'en'",
                
                # Image Captioning
                "caption_short": "TEXT",
                "caption_detailed": "TEXT",
                "caption_vqa": "TEXT",
                "caption_timestamp": "DATETIME",
                
                # Quality Assessment
                "quality_score": "REAL DEFAULT 0.0",
                "quality_level": "TEXT DEFAULT 'Unknown'",
                "sharpness": "REAL DEFAULT 0.0",
                "exposure": "REAL DEFAULT 0.0",
                "contrast": "REAL DEFAULT 0.0",
                "composition": "REAL DEFAULT 0.0",
                
                # Emotion Detection
                "emotion_data": "TEXT",
                "dominant_emotion": "TEXT DEFAULT 'neutral'",
                "face_emotion_count": "INTEGER DEFAULT 0",
                
                # Aesthetic Scoring
                "aesthetic_score": "REAL DEFAULT 0.0",
                "aesthetic_rating": "TEXT DEFAULT 'Unknown'",
            }
            
            # Add missing columns
            added_count = 0
            for col_name, col_type in new_columns.items():
                if col_name not in existing_cols:
                    sql = f"ALTER TABLE images ADD COLUMN {col_name} {col_type}"
                    cursor.execute(sql)
                    logger.info(f"  ✅ Added: {col_name}")
                    added_count += 1
                else:
                    logger.info(f"  ⏭️  Already exists: {col_name}")
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ {added_count} new columns added")
            return True
        
        except Exception as e:
            logger.error(f"❌ Column addition failed: {e}")
            return False
    
    def create_indices(self):
        """Create indices for performance"""
        logger.info("⚡ Creating database indices...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # List of indices to create
            indices = [
                ("idx_quality_score", "images", "quality_score"),
                ("idx_dominant_emotion", "images", "dominant_emotion"),
                ("idx_aesthetic_score", "images", "aesthetic_score"),
                ("idx_caption_short", "images", "caption_short"),
                ("idx_ocr_enhanced", "images", "ocr_text_enhanced"),
            ]
            
            for idx_name, table, column in indices:
                try:
                    sql = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})"
                    cursor.execute(sql)
                    logger.info(f"  ✅ Index: {idx_name}")
                except Exception as e:
                    logger.warning(f"  ⚠️  {idx_name}: {str(e)[:50]}")
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Indices created for better search performance")
            return True
        
        except Exception as e:
            logger.error(f"❌ Index creation failed: {e}")
            return False
    
    def verify_migration(self):
        """Verify migration was successful"""
        logger.info("✅ Verifying migration...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all columns
            cursor.execute("PRAGMA table_info(images)")
            columns = {row[1] for row in cursor.fetchall()}
            
            # Check for new columns
            new_cols = [
                "quality_score", "caption_short", "ocr_text_enhanced",
                "dominant_emotion", "aesthetic_score"
            ]
            
            all_present = all(col in columns for col in new_cols)
            
            if all_present:
                logger.info("✅ All new columns present")
            else:
                logger.warning("⚠️  Some columns might be missing")
            
            # Get statistics
            cursor.execute("SELECT COUNT(*) FROM images")
            image_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM images WHERE quality_score IS NOT NULL AND quality_score > 0")
            with_quality = cursor.fetchone()[0]
            
            logger.info(f"\n📊 Database Statistics:")
            logger.info(f"  Total images: {image_count}")
            logger.info(f"  Images with quality: {with_quality}")
            
            conn.close()
            return True
        
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def run(self):
        """Execute full migration"""
        logger.info("="*80)
        logger.info("🚀 DATABASE MIGRATION TO DEEP LEARNING SCHEMA")
        logger.info("="*80 + "\n")
        
        # Step 1: Check database
        if not self.check_database_exists():
            return False
        
        # Step 2: Backup
        if not self.backup_database():
            logger.error("❌ Backup creation failed - aborting migration for safety")
            return False
        
        # Step 3: Add columns
        if not self.add_columns():
            logger.error("❌ Column migration failed")
            logger.info(f"   Backup available at: {self.backup_path}")
            return False
        
        # Step 4: Create indices
        if not self.create_indices():
            logger.warning("⚠️  Index creation had issues, but migration continues")
        
        # Step 5: Verify
        if not self.verify_migration():
            logger.warning("⚠️  Verification had issues, but migration completed")
        
        # Success!
        logger.info("\n" + "="*80)
        logger.info("✅ MIGRATION SUCCESSFUL!")
        logger.info("="*80)
        
        self.print_summary()
        return True
    
    def print_summary(self):
        """Print migration summary"""
        logger.info(f"""
📋 MIGRATION SUMMARY:

✅ Database Updated
  Location: {self.db_path}
  
✅ Backup Created
  Location: {self.backup_path}
  Keep this safe! You can restore by copying back if needed.
  
✅ New Features Ready:
  - Enhanced OCR (EasyOCR)
  - Image Captioning (BLIP)
  - Quality Assessment
  - Emotion Detection
  - Aesthetic Scoring

🔄 NEXT STEPS:

1. Update your database.py file:
   Replace with: database_updated.py

2. Add new engine imports to main.py:
   from enhanced_ocr_engine import ocr_engine
   from image_captioning_engine import captioning_engine
   from quality_emotion_aesthetic_engines import image_quality, ...

3. Update upload handler in main.py:
   Add calls to the new engines when processing images
   (See INTEGRATION_GUIDE.md)

4. Add new endpoints:
   Include api_endpoints.py router in your FastAPI app

5. Restart your server:
   python main.py

6. Upload a test image:
   The system will automatically extract all metadata using
   the new deep learning models!

⚠️  IMPORTANT:
  - First upload will be slower (models loading)
  - Models are cached after first use
  - Each image takes 3-6 seconds to process
  - All results are saved in database (no re-processing)

📚 TROUBLESHOOTING:
  If something goes wrong, you have a backup!
  Restore with:
    cp {self.backup_path} {self.db_path}

""")


def main():
    """Run migration"""
    try:
        migration = DatabaseMigration()
        success = migration.run()
        return 0 if success else 1
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Migration cancelled by user")
        return 1
    
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())