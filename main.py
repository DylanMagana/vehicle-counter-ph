import argparse
import datetime
import shutil
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld

from config import (
    BUCKET_SECONDS,
    CLASSES,
    CONFIDENCE,
    DEFAULT_OUTPUT_DIR,
    FRAME_SKIP,
    MODEL_PATH,
    SPECIALISTS,
    SUMMARY_HEADER_H,
)
from detection import run_specialist
from reporting import VehicleRecord, write_report1, write_rpt_vehbytype
from video_io import (
    build_summary_frame,
    draw_polygon_zone,
    draw_supplement_boxes,
    footer_height,
    save_crop,
    select_polygon,
)


def _create_output_dirs(base: Path, classes: list[str]) -> dict[str, Path]:
    dirs: dict[str, Path] = {
        "bytype": base / "output_detections",
        "_all-detections": base / "output_detections" / "_all-detections",
        "report": base / "output_reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        (dirs["bytype"] / cls).mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Vehicle counting and classification")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    output_dir: Path = args.output.resolve()

    print("Please select input...")
    root = tk.Tk()
    root.withdraw()
    input_video = filedialog.askopenfilename(
        title="Select input video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")],
    )
    root.destroy()
    if not input_video:
        raise SystemExit("No video selected.")
    print("Input selected!")

    run_time = datetime.datetime.now()
    work_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    dirs = _create_output_dirs(work_dir, CLASSES)

    # ── Load models ───────────────────────────────────────────────────────────
    model = YOLOWorld(MODEL_PATH)
    model.set_classes(CLASSES)

    spec_models: list[YOLO] = []
    for spec_cfg in SPECIALISTS:
        spec_models.append(YOLO(spec_cfg.model_path))

    # ── Video setup ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_video)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read video: {input_video}")
    print("\nPlease define detection zone...")
    polygon = select_polygon(first_frame)
    print("Detection zone defined!")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    summary_h = height + SUMMARY_HEADER_H + footer_height(len(CLASSES))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    summary_writer = cv2.VideoWriter(
        str(work_dir / "summary.mp4"), fourcc, fps, (width, summary_h)
    )

    # ── Counters ──────────────────────────────────────────────────────────────
    veh_counter = 0
    counts_bytype = {c: 0 for c in CLASSES}
    crossed_ids: set[int] = set()
    bucket_counts: dict[int, int] = {}
    records: list[VehicleRecord] = []
    last_annotated: np.ndarray | None = None
    frame_idx = 0

    # ── Frame loop ────────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed_sec = frame_idx / fps

            if frame_idx % (FRAME_SKIP + 1) == 0:
                t0 = time.perf_counter()
                result = next(iter(model.track(
                    source=frame,
                    conf=CONFIDENCE,
                    persist=True,
                    stream=False,
                    verbose=False,
                )))
                infer_time = time.perf_counter() - t0

                # ── Normalise YOLOWorld detections ────────────────────────────
                if result.boxes is not None and result.boxes.id is not None:
                    w_ids  = result.boxes.id.cpu().numpy().astype(int)
                    w_xyxy = result.boxes.xyxy.cpu().numpy()
                    w_clss = result.boxes.cls.cpu().numpy().astype(int)
                    w_cnfs = result.boxes.conf.cpu().numpy()
                else:
                    w_ids  = np.array([], dtype=int)
                    w_xyxy = np.zeros((0, 4), dtype=float)
                    w_clss = np.array([], dtype=int)
                    w_cnfs = np.array([], dtype=float)

                # ── Run all specialists (override + supplement) ───────────────
                all_supp_boxes: list[np.ndarray] = []
                all_supp_cnfs: list[float] = []
                all_supp_ids: list[int] = []
                all_supp_cls_idxs: list[int] = []
                all_supp_labels: dict[int, list[np.ndarray]] = {}

                for i, (spec_cfg, spec_model) in enumerate(zip(SPECIALISTS, spec_models)):
                    w_clss, supp_boxes, supp_cnfs, supp_ids = run_specialist(
                        spec_cfg, spec_model, frame, CLASSES, w_xyxy, w_clss,
                        id_offset=(i + 1) * 10_000_000,
                    )
                    cls_idx = CLASSES.index(spec_cfg.class_name)
                    all_supp_boxes.extend(supp_boxes)
                    all_supp_cnfs.extend(supp_cnfs)
                    all_supp_ids.extend(supp_ids)
                    all_supp_cls_idxs.extend([cls_idx] * len(supp_boxes))
                    all_supp_labels.setdefault(cls_idx, []).extend(supp_boxes)

                # ── Merge world + supplements ─────────────────────────────────
                all_ids  = list(w_ids)  + all_supp_ids
                all_xyxy = list(w_xyxy) + all_supp_boxes
                all_clss = list(w_clss) + all_supp_cls_idxs
                all_cnfs = list(w_cnfs) + all_supp_cnfs

                for track_id, box, cls_idx, conf in zip(all_ids, all_xyxy, all_clss, all_cnfs):
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                    inside = cv2.pointPolygonTest(polygon, center, False) >= 0
                    if inside and track_id not in crossed_ids:
                        crossed_ids.add(track_id)
                        veh_counter += 1
                        label = CLASSES[cls_idx]
                        counts_bytype[label] += 1

                        fname = f"veh_{veh_counter:05d}.png"
                        save_crop(
                            frame, box, width, height,
                            [dirs["bytype"] / label, dirs["_all-detections"]],
                            fname,
                        )

                        bucket = int(elapsed_sec // BUCKET_SECONDS) * BUCKET_SECONDS
                        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

                        final_path = output_dir / (dirs["bytype"] / label / fname).relative_to(work_dir)
                        records.append(VehicleRecord(
                            filename=str(final_path),
                            processingtime=infer_time,
                            label=label,
                            score=float(conf),
                        ))

                # ── Annotate frame ────────────────────────────────────────────
                annotated = result.plot()
                for cls_idx, boxes in all_supp_labels.items():
                    draw_supplement_boxes(annotated, boxes, CLASSES[cls_idx])
                last_annotated = draw_polygon_zone(annotated, polygon)

            if last_annotated is not None:
                summary_frame = build_summary_frame(
                    last_annotated, counts_bytype, run_time, input_video, str(output_dir),
                )
                summary_writer.write(summary_frame)

            frame_idx += 1
            pct = frame_idx / total if total > 0 else 0.0
            filled = int(40 * pct)
            bar = "❚" * filled + " " * (40 - filled)
            print(f"\r{frame_idx}/{total} frames processed ; [{bar}] {pct:5.1%} ", end="", flush=True)
    finally:
        cap.release()
        summary_writer.release()
    print()

    # ── Reports ───────────────────────────────────────────────────────────────
    total_secs = int(total / fps)
    write_report1(dirs["report"] / "vehicle_counts_by_time.csv", bucket_counts, total_secs, BUCKET_SECONDS)
    write_rpt_vehbytype(dirs["report"] / "vehicle_detections.csv", records)

    # ── Swap temp dir → final output ──────────────────────────────────────────
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.move(str(work_dir), str(output_dir))

    print(f"\nVehicles counted: {veh_counter}")
    for cls, count in counts_bytype.items():
        print(f"-> {cls}: {count}")
    print(f"\nOutput saved to: {output_dir}\n\n")


if __name__ == "__main__":
    main()
