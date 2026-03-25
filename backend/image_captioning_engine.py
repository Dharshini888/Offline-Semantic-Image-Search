"""
Image Captioning Engine — 100% Offline using BLIP
===================================================

HOW OFFLINE MODE WORKS
-----------------------
1. FIRST RUN (internet required, one-time only):
   The model is downloaded from HuggingFace and saved to:
       ../models/blip-image-captioning-base/
   This is ~1 GB and happens automatically on the very first call.

2. ALL SUBSEQUENT RUNS (fully offline):
   The model is loaded directly from the local folder — no internet needed.
   Setting TRANSFORMERS_OFFLINE=1 and local_files_only=True enforces this.

3. ZERO-INTERNET MACHINES:
   Copy the folder  ../models/blip-image-captioning-base/  from any machine
   that ran the first-time download, and it will work permanently offline.

Run the download manually once:
    python image_captioning_engine.py --download

Usage:
    from image_captioning_engine import captioning_engine
    caption = captioning_engine.generate_caption("photo.jpg")
"""

import os
import sys
import logging
import torch
from PIL import Image
from pathlib import Path

logger = logging.getLogger("ImageCaptioningEngine")

# ── Paths ────────────────────────────────────────────────────────────────────
# Resolve relative to THIS file so it works no matter where you call from.
_HERE = Path(__file__).resolve().parent
MODEL_LOCAL_DIR = str(_HERE / ".." / "models" / "blip-image-captioning-base")
MODEL_HF_NAME   = "Salesforce/blip-image-captioning-base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Lazy imports — avoid import-time crashes if transformers not installed ────
BlipProcessor              = None
BlipForConditionalGeneration = None

def _import_transformers():
    global BlipProcessor, BlipForConditionalGeneration
    if BlipProcessor is not None:
        return True
    try:
        from transformers import BlipProcessor as BP, BlipForConditionalGeneration as BM
        BlipProcessor = BP
        BlipForConditionalGeneration = BM
        return True
    except ImportError:
        logger.error("transformers not installed. Run: pip install transformers")
        return False


# ── Download helper ──────────────────────────────────────────────────────────
def download_blip_model(force: bool = False):
    """
    Download BLIP model from HuggingFace and save it locally.
    Safe to call multiple times — skips if already present.

    Args:
        force: re-download even if the local folder exists
    """
    if not _import_transformers():
        return False

    local = Path(MODEL_LOCAL_DIR)

    # Already downloaded?
    marker = local / "config.json"
    if marker.exists() and not force:
        logger.info(f"✅ BLIP model already cached at: {local}")
        return True

    logger.info(f"⬇️  Downloading BLIP model from HuggingFace → {local}")
    logger.info("   (This is a one-time ~1 GB download — please wait...)")

    try:
        local.mkdir(parents=True, exist_ok=True)

        # Download processor
        proc = BlipProcessor.from_pretrained(MODEL_HF_NAME)
        proc.save_pretrained(str(local))
        logger.info("   ✓ Processor saved")

        # Download model weights
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = BlipForConditionalGeneration.from_pretrained(
            MODEL_HF_NAME,
            torch_dtype=dtype,
        )
        model.save_pretrained(str(local))
        logger.info("   ✓ Model weights saved")

        logger.info(f"✅ Download complete → {local}")
        return True

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


# ── Main Engine ──────────────────────────────────────────────────────────────
class ImageCaptioningEngine:
    """
    BLIP image captioning — loads 100% from local disk after first download.

    Public API
    ----------
    generate_caption(image_path, max_length=30)          → str
    generate_conditional_caption(image_path, condition)  → str
    answer_visual_question(image_path, question)         → str
    generate_detailed_description(image_path)            → dict
    batch_caption_images(image_paths)                    → list[dict]
    """

    def __init__(self):
        self.processor = None
        self.model     = None
        self._loaded   = False
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self):
        """
        Try to load from local folder first.
        If the folder is missing, attempt a one-time internet download.
        """
        if not _import_transformers():
            return

        local = Path(MODEL_LOCAL_DIR)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # ── 1. Load from local disk (offline, instant) ────────────────────────
        if (local / "config.json").exists():
            try:
                logger.info(f"📂 Loading BLIP from local cache: {local}")
                self.processor = BlipProcessor.from_pretrained(
                    str(local),
                    local_files_only=True,
                )
                self.model = BlipForConditionalGeneration.from_pretrained(
                    str(local),
                    torch_dtype=dtype,
                    local_files_only=True,
                ).to(DEVICE)
                self.model.eval()
                self._loaded = True
                logger.info(f"✅ BLIP loaded offline on {DEVICE}")
                return
            except Exception as e:
                logger.warning(f"⚠️  Local load failed ({e}), trying download...")

        # ── 2. First-time: download and then load ─────────────────────────────
        logger.info("🌐 Local model not found — downloading once (needs internet)...")
        if download_blip_model():
            try:
                self.processor = BlipProcessor.from_pretrained(
                    str(local),
                    local_files_only=True,
                )
                self.model = BlipForConditionalGeneration.from_pretrained(
                    str(local),
                    torch_dtype=dtype,
                    local_files_only=True,
                ).to(DEVICE)
                self.model.eval()
                self._loaded = True
                logger.info(f"✅ BLIP ready on {DEVICE} (loaded after download)")
            except Exception as e:
                logger.error(f"❌ Model load after download failed: {e}")
        else:
            logger.error(
                "❌ BLIP model unavailable.\n"
                "   Run once with internet:  python image_captioning_engine.py --download\n"
                f"  Then copy the folder to: {local}"
            )

    def _is_ready(self) -> bool:
        if not self._loaded or self.model is None or self.processor is None:
            logger.warning("⚠️  BLIP not loaded — caption unavailable.")
            return False
        return True

    def _prepare_inputs(self, image_path: str, text: str = None):
        """Open image and build processor inputs. Returns None on failure."""
        try:
            img = Image.open(image_path).convert("RGB")
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            if text:
                inputs = self.processor(img, text, return_tensors="pt")
            else:
                inputs = self.processor(img, return_tensors="pt")
            return {k: v.to(DEVICE, dtype) if v.dtype.is_floating_point else v.to(DEVICE)
                    for k, v in inputs.items()}
        except Exception as e:
            logger.error(f"Image open/preprocess failed for {image_path}: {e}")
            return None

    # ── Public methods ────────────────────────────────────────────────────────

    def generate_caption(self, image_path: str, max_length: int = 30) -> str:
        """
        Generate a short natural-language caption for an image.

        Args:
            image_path : path to a JPG/PNG/WEBP image
            max_length : token budget for the output (default 30 ≈ one sentence)

        Returns:
            Descriptive caption string, or "" if model unavailable / image broken.

        Example:
            >>> captioning_engine.generate_caption("holiday.jpg")
            'a family sitting on a beach with a sunset in the background'
        """
        if not self._is_ready():
            return ""

        inputs = self._prepare_inputs(image_path)
        if inputs is None:
            return ""

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=4,
                    early_stopping=True,
                    repetition_penalty=1.3,   # avoids repeating words
                )
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption.strip()
        except Exception as e:
            logger.error(f"Caption generation failed for {image_path}: {e}")
            return ""

    def generate_conditional_caption(
        self,
        image_path: str,
        condition: str = "",
        max_length: int = 50,
    ) -> str:
        """
        Generate a caption guided by an optional text condition/prefix.

        Args:
            image_path : path to image
            condition  : text prefix, e.g. "a photo of a person wearing"
            max_length : max output tokens

        Returns:
            Caption string.

        Example:
            >>> captioning_engine.generate_conditional_caption("pic.jpg", "the people in this photo are")
            'the people in this photo are smiling and wearing colorful clothes'
        """
        if not self._is_ready():
            return ""

        inputs = self._prepare_inputs(image_path, condition if condition else None)
        if inputs is None:
            return ""

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=4,
                    early_stopping=True,
                    repetition_penalty=1.3,
                )
            return self.processor.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Conditional caption failed for {image_path}: {e}")
            return ""

    def answer_visual_question(self, image_path: str, question: str) -> str:
        """
        Visual Question Answering (VQA) — ask any question about an image.

        Args:
            image_path : path to image
            question   : natural-language question

        Returns:
            Short answer string.

        Examples:
            >>> captioning_engine.answer_visual_question("photo.jpg", "What color is the car?")
            'red'
            >>> captioning_engine.answer_visual_question("photo.jpg", "Is there a person?")
            'yes'
        """
        if not self._is_ready():
            return ""

        inputs = self._prepare_inputs(image_path, question)
        if inputs is None:
            return ""

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    num_beams=4,
                )
            return self.processor.decode(out[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"VQA failed for {image_path}: {e}")
            return ""

    def generate_detailed_description(self, image_path: str) -> dict:
        """
        Generate a rich, structured description by running multiple BLIP passes.

        Returns a dict with keys:
            short_caption   – one sentence (~10 words)
            detailed_caption – longer natural description
            visual_qa       – dict of question → answer pairs
            searchable_text – all text joined for full-text search indexing

        Compatible with the fields used in main.py upload handler:
            img_record.caption_short    = result["short_caption"]
            img_record.caption_detailed = result["detailed_caption"]
        """
        if not self._is_ready():
            return {}

        short    = self.generate_caption(image_path, max_length=20)
        detailed = self.generate_caption(image_path, max_length=60)

        vqa = {}
        questions = [
            ("What is the main subject?",          "subject"),
            ("What color is the main object?",     "color"),
            ("Is there a person in the image?",    "has_person"),
            ("What is the setting or location?",   "location"),
            ("What activity is happening?",        "activity"),
        ]
        for q_text, q_key in questions:
            try:
                ans = self.answer_visual_question(image_path, q_text)
                if ans:
                    vqa[q_key] = ans
            except Exception:
                pass

        searchable = " ".join(filter(None, [short, detailed] + list(vqa.values())))

        return {
            "short_caption":    short,
            "detailed_caption": detailed,
            "visual_qa":        vqa,
            "searchable_text":  searchable,
        }

    def batch_caption_images(self, image_paths: list, log_every: int = 10) -> list:
        """
        Caption a list of images, returning a list of result dicts.

        Args:
            image_paths : list of file path strings
            log_every   : log progress every N images

        Returns:
            [{"image": path, "caption": "..."}, ...]
        """
        results = []
        for i, path in enumerate(image_paths, 1):
            caption = self.generate_caption(path)
            results.append({"image": path, "caption": caption})
            if i % log_every == 0:
                logger.info(f"  Captioned {i}/{len(image_paths)}")
        return results


# ── Module-level singleton ─────────────────────────────────────────────────
captioning_engine = ImageCaptioningEngine()


# ── CLI helper ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    if "--download" in sys.argv:
        # python image_captioning_engine.py --download
        print(f"\nDownloading BLIP model to: {MODEL_LOCAL_DIR}")
        ok = download_blip_model(force="--force" in sys.argv)
        sys.exit(0 if ok else 1)

    if len(sys.argv) >= 2 and not sys.argv[1].startswith("--"):
        # python image_captioning_engine.py path/to/image.jpg
        img_arg = sys.argv[1]
        if not os.path.exists(img_arg):
            print(f"File not found: {img_arg}")
            sys.exit(1)

        engine = ImageCaptioningEngine()
        print("\n── Short caption ──────────────────────────────────────────")
        print(engine.generate_caption(img_arg))
        print("\n── Detailed description ───────────────────────────────────")
        desc = engine.generate_detailed_description(img_arg)
        for k, v in desc.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    print(__doc__)
    print("\nUsage:")
    print("  python image_captioning_engine.py --download        # one-time download")
    print("  python image_captioning_engine.py photo.jpg         # caption an image")