import numpy as np

from config import SpecialistConfig


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1 = max(xa1, xb1)
    iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2)
    iy2 = min(ya2, yb2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter)


def run_specialist(
    spec_cfg: SpecialistConfig,
    model: object,
    frame: np.ndarray,
    classes: list[str],
    w_xyxy: np.ndarray,
    w_clss: np.ndarray,
    id_offset: int = 0,
) -> tuple[np.ndarray, list[np.ndarray], list[float], list[int]]:
    """Run one specialist model via ByteTrack and return (updated w_clss, supp_boxes, supp_confs, supp_ids).

    Uses model.track() with persist=True so each specialist maintains its own ByteTrack state.
    Track IDs are shifted by id_offset to avoid collision with YOLOWorld's ByteTrack IDs.
    Mutates ``w_clss`` in-place (relabels overlapping world boxes).
    """
    result = model.track(source=frame, conf=spec_cfg.confidence, persist=True, verbose=False)[0]
    cls_idx = classes.index(spec_cfg.class_name)

    supp_boxes: list[np.ndarray] = []
    supp_cnfs: list[float] = []
    supp_ids: list[int] = []

    if result.boxes is not None and len(result.boxes):
        spec_xyxy = result.boxes.xyxy.cpu().numpy()
        spec_cnfs = result.boxes.conf.cpu().numpy()
        spec_ids = (
            result.boxes.id.cpu().numpy().astype(int)
            if result.boxes.id is not None
            else np.arange(1, len(spec_xyxy) + 1, dtype=int)
        )

        # Override: relabel world boxes that overlap a specialist box
        for i in range(len(w_xyxy)):
            best = max((iou(w_xyxy[i], sb) for sb in spec_xyxy), default=0.0)
            if best >= spec_cfg.iou_merge:
                w_clss[i] = cls_idx

        # Supplement: specialist boxes with no world overlap
        for sbox, sconf, sid in zip(spec_xyxy, spec_cnfs, spec_ids):
            max_overlap = max((iou(sbox, wb) for wb in w_xyxy), default=0.0)
            if max_overlap < spec_cfg.iou_merge:
                supp_boxes.append(sbox)
                supp_cnfs.append(float(sconf))
                supp_ids.append(id_offset + int(sid))

    return w_clss, supp_boxes, supp_cnfs, supp_ids
