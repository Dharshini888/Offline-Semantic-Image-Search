import os
import sqlite3
import sys

print("\n" + "=" * 70)
print("📊 OFFLINE PHOTO GALLERY - DIAGNOSTIC REPORT")
print("=" * 70)

BASE_DIR = ".."
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "db.sqlite")
FAISS_IMAGE_PATH = os.path.join(DATA_DIR, "index.faiss")
FAISS_FACE_PATH = os.path.join(DATA_DIR, "face_index.faiss")

# ============================================================================
# CHECK 1: Image Directory
# ============================================================================
print("\n1️⃣  IMAGE DIRECTORY")
print("-" * 70)

if not os.path.exists(IMAGE_DIR):
    print(f"❌ Image directory NOT found: {os.path.abspath(IMAGE_DIR)}")
    print("   Create it and add photos!")
    sys.exit(1)

print(f"✅ Image directory exists: {os.path.abspath(IMAGE_DIR)}")

# List all files
all_files = os.listdir(IMAGE_DIR)
print(f"   Total files: {len(all_files)}")

# Filter images
valid_images = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"   Valid images (JPG/PNG): {len(valid_images)}")

if len(valid_images) == 0:
    print("\n❌ PROBLEM: No JPG/PNG images found!")
    print("\n   Files present:")
    for f in all_files[:20]:
        print(f"   - {f}")
    if len(all_files) > 20:
        print(f"   ... and {len(all_files) - 20} more")
    print("\n   Solution: Add JPG or PNG image files to this folder")
    sys.exit(1)

print(f"\n   Sample images:")
for img in valid_images[:5]:
    path = os.path.join(IMAGE_DIR, img)
    size = os.path.getsize(path)
    print(f"   ✓ {img} ({size:,} bytes)")
if len(valid_images) > 5:
    print(f"   ... and {len(valid_images) - 5} more")

# ============================================================================
# CHECK 2: Database
# ============================================================================
print("\n2️⃣  DATABASE")
print("-" * 70)

if not os.path.exists(DB_PATH):
    print(f"❌ Database NOT found: {os.path.abspath(DB_PATH)}")
    print("   Solution: Run 'python build_index.py'")
    sys.exit(1)

print(f"✅ Database exists: {os.path.abspath(DB_PATH)}")

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check images
    cursor.execute("SELECT COUNT(*) FROM images")
    img_count = cursor.fetchone()[0]
    
    # Check faces
    cursor.execute("SELECT COUNT(*) FROM faces")
    face_count = cursor.fetchone()[0]
    
    # Check people
    cursor.execute("SELECT COUNT(*) FROM people")
    people_count = cursor.fetchone()[0]
    
    # Check albums
    cursor.execute("SELECT COUNT(*) FROM albums")
    album_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"   Images in DB: {img_count}")
    print(f"   Faces found: {face_count}")
    print(f"   People (clusters): {people_count}")
    print(f"   Albums/Events: {album_count}")
    
    if img_count == 0:
        print(f"\n❌ PROBLEM: No images in database!")
        print("   Solution:")
        print("   1. Delete db.sqlite: del ..\\data\\db.sqlite")
        print("   2. Re-run: python build_index.py")
        print("   3. Wait for it to complete")
        sys.exit(1)
    
except Exception as e:
    print(f"❌ Error reading database: {e}")
    sys.exit(1)

# ============================================================================
# CHECK 3: FAISS Indexes
# ============================================================================
print("\n3️⃣  FAISS INDEXES")
print("-" * 70)

if not os.path.exists(FAISS_IMAGE_PATH):
    print(f"❌ Image FAISS index NOT found: {os.path.abspath(FAISS_IMAGE_PATH)}")
    print("   Solution:")
    print("   1. Delete: del ..\\data\\db.sqlite")
    print("   2. Re-run: python build_index.py")
else:
    size = os.path.getsize(FAISS_IMAGE_PATH)
    print(f"✅ Image FAISS index: {size:,} bytes")

if not os.path.exists(FAISS_FACE_PATH):
    print(f"❌ Face FAISS index NOT found: {os.path.abspath(FAISS_FACE_PATH)}")
    if face_count > 0:
        print("   Solution:")
        print("   1. Delete: del ..\\data\\face_index.faiss")
        print("   2. Re-run: python build_index.py")
else:
    size = os.path.getsize(FAISS_FACE_PATH)
    print(f"✅ Face FAISS index: {size:,} bytes")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)

if img_count > 0 and os.path.exists(FAISS_IMAGE_PATH):
    print("\n✅ Everything looks good!")
    print(f"   • {img_count} images indexed")
    print(f"   • {face_count} faces detected")
    print(f"   • {people_count} people (face clusters)")
    print(f"   • {album_count} albums/events")
    print("\n   Next step: Open http://localhost:8000 in your browser")
    print("   If server not running: python main.py")
elif img_count == 0:
    print("\n❌ Images not indexed")
    print("   Solution:")
    print("   1. Make sure JPG/PNG files are in ../data/images/")
    print("   2. Delete old data: del ..\\data\\db.sqlite")
    print("   3. Rebuild: python build_index.py")
    print("   4. Start server: python main.py")
elif not os.path.exists(FAISS_IMAGE_PATH):
    print("\n❌ FAISS index not created")
    print("   Solution:")
    print("   1. Delete old data: del ..\\data\\db.sqlite")
    print("   2. Rebuild: python build_index.py")
    print("   3. Wait for it to complete")
    print("   4. Start server: python main.py")

print("\n" + "=" * 70 + "\n")
