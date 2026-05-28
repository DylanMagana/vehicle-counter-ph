from dataclasses import dataclass
from pathlib import Path


# ── Paths & general ───────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = Path("outputs/sample")
MODEL_PATH = "models/yolov8l-worldv2.pt"
CLASSES: list[str] = ["car", "van", "truck", "bus", "taxi", "motorcycle", "jeepney", "tricycle", "ebike"]

# ── Inference ─────────────────────────────────────────────────────────────────
FRAME_SKIP = 4        # infer every (FRAME_SKIP + 1)th frame; 0 = every frame
CONFIDENCE = 0.45

# ── Specialist models ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SpecialistConfig:
    model_path: str
    class_name: str
    confidence: float = 0.35
    iou_merge: float = 0.1


SPECIALISTS: list[SpecialistConfig] = [
    SpecialistConfig(
        model_path="models/yolov11m-jeepney.pt",
        class_name="jeepney",
        confidence=0.5,
        iou_merge=0.1,
    ),
    SpecialistConfig(
        model_path="models/yolov11m-ebike2.pt",
        class_name="ebike",
        confidence=0.5,
        iou_merge=0.1,
    ),
]

# ── Reporting ─────────────────────────────────────────────────────────────────
BUCKET_SECONDS = 1     # time-bucket width for report1.csv

# ── Summary video layout ─────────────────────────────────────────────────────
SUMMARY_HEADER_H = 110        # pixel height of the header block
SUMMARY_FOOTER_ROW_H = 70     # pixel height of each row in the footer
SUMMARY_MAX_COLS = 5           # max class columns per row in the footer
