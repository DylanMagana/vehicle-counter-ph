import math

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


class CentroidTracker:
    """Minimal centroid tracker for specialist supplement detections.

    Track IDs start at an offset to avoid collision with YOLOWorld's ByteTrack IDs.
    Each instance gets its own ID range (10M apart).
    """

    _instance_count = 0

    def __init__(self, max_disappeared: int = 15, max_distance: float = 80.0) -> None:
        self._next_id = 10_000_000 + CentroidTracker._instance_count * 10_000_000
        CentroidTracker._instance_count += 1
        self._objects: dict[int, tuple[float, float]] = {}
        self._disappeared: dict[int, int] = {}
        self._max_disappeared = max_disappeared
        self._max_distance = max_distance

    def update(self, centroids: list[tuple[float, float]]) -> list[int]:
        for oid in list(self._objects):
            self._disappeared[oid] = self._disappeared.get(oid, 0) + 1
            if self._disappeared[oid] > self._max_disappeared:
                del self._objects[oid]
                del self._disappeared[oid]

        if not centroids:
            return []

        assigned: list[int] = []
        used: set[int] = set()

        for cx, cy in centroids:
            best_oid, best_dist = None, float("inf")
            for oid, (ox, oy) in self._objects.items():
                if oid in used:
                    continue
                d = math.hypot(cx - ox, cy - oy)
                if d < best_dist:
                    best_dist = d
                    best_oid = oid

            if best_oid is not None and best_dist < self._max_distance:
                self._objects[best_oid] = (cx, cy)
                self._disappeared[best_oid] = 0
                assigned.append(best_oid)
                used.add(best_oid)
            else:
                new_id = self._next_id
                self._next_id += 1
                self._objects[new_id] = (cx, cy)
                self._disappeared[new_id] = 0
                assigned.append(new_id)

        return assigned


def run_specialist(
    spec_cfg: SpecialistConfig,
    model: object,
    tracker: CentroidTracker,
    frame: np.ndarray,
    classes: list[str],
    w_xyxy: np.ndarray,
    w_clss: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], list[float], list[int]]:
    """Run one specialist model and return (updated w_clss, supp_boxes, supp_confs, supp_ids).

    Mutates ``w_clss`` in-place (relabels overlapping world boxes).
    """
    result = model.predict(source=frame, conf=spec_cfg.confidence, verbose=False)[0]
    cls_idx = classes.index(spec_cfg.class_name)

    supp_boxes: list[np.ndarray] = []
    supp_cnfs: list[float] = []

    if result.boxes is not None and len(result.boxes):
        spec_xyxy = result.boxes.xyxy.cpu().numpy()
        spec_cnfs = result.boxes.conf.cpu().numpy()

        # Override: relabel world boxes that overlap a specialist box
        for i in range(len(w_xyxy)):
            best = max((iou(w_xyxy[i], sb) for sb in spec_xyxy), default=0.0)
            if best >= spec_cfg.iou_merge:
                w_clss[i] = cls_idx

        # Supplement: specialist boxes with no world overlap
        for sbox, sconf in zip(spec_xyxy, spec_cnfs):
            max_overlap = max((iou(sbox, wb) for wb in w_xyxy), default=0.0)
            if max_overlap < spec_cfg.iou_merge:
                supp_boxes.append(sbox)
                supp_cnfs.append(float(sconf))

    centroids = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in supp_boxes]
    supp_ids = tracker.update(centroids)

    return w_clss, supp_boxes, supp_cnfs, supp_ids
