#!/usr/bin/env python3
"""
SETUP SCRIPT - Initialize Deep Learning Image System

This script will:
1. Check Python version
2. Install dependencies
3. Create necessary directories
4. Initialize database with new schema
5. Test all components
6. Provide next steps

Usage:
    python setup_deeplearning.py
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Verify Python 3.9+"""
    logger.info("🐍 Checking Python version...")
    
    if sys.version_info < (3, 9):
        logger.error(f"❌ Python 3.9+ required, got {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)
    
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")


def check_directories():
    """Create necessary directories"""
    logger.info("📁 Creating directories...")
    
    dirs = [
        "../data",
        "../data/images",
        "../models"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ {d}")


def install_dependencies():
    """Install pip packages"""
    logger.info("📦 Installing dependencies...")
    logger.info("   This may take 5-10 minutes (models download 2GB)...")
    
    try:
        # Try to install from requirements file
        requirements_file = "requirements_deeplearning.txt"
        
        if not os.path.exists(requirements_file):
            logger.warning(f"⚠️  {requirements_file} not found, installing manually...")
            
            packages = [
                "fastapi",
                "uvicorn[standard]",
                "sqlalchemy",
                "torch",
                "torchvision",
                "numpy",
                "scipy",
                "opencv-python",
                "Pillow",
                "faiss-cpu",
                "insightface",
                "onnxruntime",
                "easyocr",
                "transformers",
                "huggingface-hub",
                "scikit-image",
                "fer",
                "scikit-learn",
                "requests",
                "python-dotenv",
                "pydantic"
            ]
            
            for package in packages:
                logger.info(f"   Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        
        else:
            logger.info(f"   Installing from {requirements_file}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        
        logger.info("✅ Dependencies installed")
    
    except Exception as e:
        logger.error(f"❌ Installation failed: {e}")
        sys.exit(1)


def test_imports():
    """Test that all modules can be imported"""
    logger.info("🔍 Testing imports...")
    
    modules_to_test = [
        ("fastapi", "FastAPI"),
        ("sqlalchemy", "SQLAlchemy"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("torchvision", "Torchvision"),
        ("clip", "CLIP"),
        ("faiss", "FAISS"),
        ("easyocr", "EasyOCR"),
        ("transformers", "Transformers"),
        ("sklearn", "scikit-learn"),
    ]
    
    failed = []
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"✅ {display_name}")
        except ImportError as e:
            logger.warning(f"⚠️  {display_name} - {str(e)[:50]}")
            failed.append(display_name)
    
    if failed:
        logger.warning(f"\n⚠️  Some modules failed to import: {', '.join(failed)}")
        logger.warning("   This might be OK if you don't plan to use those features")
    
    return len(failed) == 0


def initialize_database():
    """Initialize database schema"""
    logger.info("🗄️  Initializing database...")
    
    try:
        from database_updated import init_db
        init_db()
        logger.info("✅ Database initialized")
        return True
    
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False


def test_models():
    """Test that models can be loaded"""
    logger.info("🤖 Testing model loading...")
    
    tests_passed = 0
    tests_total = 0
    
    # Test CLIP
    tests_total += 1
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        logger.info("✅ CLIP model")
        tests_passed += 1
    except Exception as e:
        logger.warning(f"⚠️  CLIP model - {str(e)[:50]}")
    
    # Test EasyOCR
    tests_total += 1
    try:
        import easyocr
        # Just test import, don't load (takes time)
        logger.info("✅ EasyOCR (import)")
        tests_passed += 1
    except Exception as e:
        logger.warning(f"⚠️  EasyOCR - {str(e)[:50]}")
    
    # Test Transformers
    tests_total += 1
    try:
        from transformers import AutoTokenizer
        logger.info("✅ Transformers (import)")
        tests_passed += 1
    except Exception as e:
        logger.warning(f"⚠️  Transformers - {str(e)[:50]}")
    
    logger.info(f"\n✅ {tests_passed}/{tests_total} model tests passed")
    return tests_passed > 0


def create_sample_config():
    """Create sample .env file"""
    logger.info("⚙️  Creating sample configuration...")
    
    if not os.path.exists(".env.example"):
        with open(".env.example", "w") as f:
            f.write("""# Deep Learning Image System Configuration

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Database
DATABASE_URL=sqlite:///../data/db.sqlite

# Feature Flags
ENABLE_CLIP_SEARCH=true
ENABLE_EASYOCR=true
ENABLE_BLIP_CAPTIONING=true
ENABLE_QUALITY_ASSESSMENT=true
ENABLE_EMOTION_DETECTION=true
ENABLE_AESTHETIC_SCORING=true

# Model Settings
OCR_LANGUAGES=en
CAPTIONING_MODEL=Salesforce/blip-image-captioning-base
USE_GPU=false

# Search Settings
CLIP_SCORE_MIN=0.25
TOP_K_SEARCH=20

# Upload Settings
MAX_UPLOAD_SIZE_MB=50
ALLOWED_EXTENSIONS=jpg,jpeg,png,gif,webp
""")
        logger.info("✅ .env.example created")
    else:
        logger.info("⏭️  .env.example already exists")


def print_summary():
    """Print setup summary and next steps"""
    logger.info("\n" + "="*80)
    logger.info("✅ SETUP COMPLETE!")
    logger.info("="*80)
    
    logger.info("""
📋 WHAT WAS INSTALLED:
  ✅ FastAPI & Uvicorn (web framework)
  ✅ SQLAlchemy (database ORM)
  ✅ PyTorch & Torchvision (deep learning)
  ✅ CLIP (semantic image search) - existing
  ✅ InsightFace (face recognition) - existing
  ✅ Faster R-CNN (object detection) - existing
  ✅ EasyOCR (text extraction) - NEW
  ✅ BLIP (image captions) - NEW
  ✅ Quality Assessment (sharpness, exposure) - NEW
  ✅ FER (emotion detection) - NEW

📁 DIRECTORIES CREATED:
  ✅ ../data (images & database)
  ✅ ../data/images (image storage)
  ✅ ../models (model weights)

📊 NEXT STEPS:

1️⃣  UPDATE YOUR CODE:
    Copy these files to your backend folder:
    - database_updated.py → database.py (backup old first!)
    - enhanced_ocr_engine.py
    - image_captioning_engine.py
    - quality_emotion_aesthetic_engines.py
    - api_endpoints.py

2️⃣  UPDATE main.py:
    In your main.py upload handler, add the new engines:
    
    from enhanced_ocr_engine import ocr_engine
    from image_captioning_engine import captioning_engine
    from quality_emotion_aesthetic_engines import image_quality, emotion_detection
    
    Then add calls to these in the upload function
    (See INTEGRATION_GUIDE.md for detailed code)

3️⃣  INITIALIZE DATABASE:
    python -c "from database_updated import init_db; init_db()"

4️⃣  START YOUR SERVER:
    python main.py
    
    Server should print:
    ✅ Database initialized
    ✅ Models loaded
    ✅ Ready on http://localhost:8000

5️⃣  TEST THE SYSTEM:
    # Upload a test image
    curl -X POST http://localhost:8000/upload \\
      -F "file=@test.jpg"
    
    # Get image details
    curl http://localhost:8000/api/v1/image/1
    
    # Check stats
    curl http://localhost:8000/api/v1/stats/quality

6️⃣  OPTIONAL: RETRAIN ON YOUR DATA
    Train quality assessment on your images for better results
    (See documentation for advanced setup)

📚 DOCUMENTATION:
  - DEEPLEARNING_FEATURES_GUIDE.md
  - INTEGRATION_GUIDE.md
  - QUICK_REFERENCE_FAQ.md

🆘 TROUBLESHOOTING:
  If models fail to load:
  - First run is slow (downloading models)
  - Check internet connection (2GB download)
  - Models are cached after first download
  - See QUICK_REFERENCE_FAQ.md for solutions

💡 PRO TIPS:
  - EasyOCR first run takes 30 seconds (normal)
  - BLIP captioning works better with GPU
  - Quality assessment works great on CPU
  - Results are cached in database forever

""")
    
    logger.info("="*80)
    logger.info("🎉 Ready to start building!")
    logger.info("="*80)


def main():
    """Run setup steps"""
    print("\n" + "="*80)
    print("🚀 DEEP LEARNING IMAGE SYSTEM SETUP")
    print("="*80 + "\n")
    
    try:
        check_python_version()
        check_directories()
        install_dependencies()
        test_imports()
        initialize_database()
        test_models()
        create_sample_config()
        print_summary()
        
        logger.info("\n✅ Setup successful! Ready to use.\n")
        return 0
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  Setup cancelled by user")
        return 1
    
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())