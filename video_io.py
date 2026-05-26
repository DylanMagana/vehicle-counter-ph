import datetime
import math
from pathlib import Path

import cv2
import numpy as np

from config import SUMMARY_FOOTER_ROW_H, SUMMARY_HEADER_H, SUMMARY_MAX_COLS


def select_polygon(frame: np.ndarray) -> np.ndarray:
    """Show the first video frame and let the user click polygon vertices.

    Controls:
      Left-click  — add a vertex
      R           — reset all vertices
      Enter/Space — confirm (requires >= 3 points)
      ESC         — abort and exit the program
    """
    points: list[tuple[int, int]] = []
    display = frame.copy()
    win = "Define counting zone  |  click=add point  R=reset  Enter=confirm  ESC=quit"
    cv2.namedWindow(win)

    def _redraw() -> None:
        nonlocal display
        display = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display, points[i - 1], pt, (0, 255, 0), 2)
        if len(points) > 2:
            cv2.line(display, points[-1], points[0], (0, 180, 0), 1)
        cv2.imshow(win, display)

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            _redraw()

    cv2.setMouseCallback(win, _on_mouse)
    cv2.imshow(win, display)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter or Space
            if len(points) >= 3:
                break
        elif key == ord("r"):
            points.clear()
            _redraw()
        elif key == 27:  # ESC
            cv2.destroyWindow(win)
            raise SystemExit("Polygon selection cancelled.")

    cv2.destroyWindow(win)
    return np.array(points, dtype=np.int32)


def footer_height(num_classes: int) -> int:
    rows = max(1, math.ceil(num_classes / SUMMARY_MAX_COLS))
    return rows * SUMMARY_FOOTER_ROW_H


def draw_polygon_zone(frame: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], (0, 255, 0))
    result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    cv2.polylines(result, [polygon], isClosed=True, color=(0, 200, 0), thickness=2)
    return result


def draw_supplement_boxes(
    annotated: np.ndarray,
    supp_boxes: list[np.ndarray],
    label: str,
) -> None:
    for sbox in supp_boxes:
        bx1, by1, bx2, by2 = map(int, sbox)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
        cv2.putText(
            annotated, f"{label}*", (bx1, by1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA,
        )


def build_summary_frame(
    frame: np.ndarray,
    counts: dict[str, int],
    run_time: datetime.datetime,
    input_path: str,
    output_path: str,
) -> np.ndarray:
    w = frame.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX

    header = np.ones((SUMMARY_HEADER_H, w, 3), dtype=np.uint8) * 255
    header_lines: list[tuple[str, tuple[int, int], float, int]] = [
        ("ARUP Manila - Vehicle Counting and Classification Model", (10, 25), 0.55, 1),
        (f"Run Time Record: {run_time.strftime('%Y-%m-%d, %H:%M:%S')}", (10, 50), 0.45, 1),
        (f"Input video directory: {input_path}", (10, 70), 0.40, 1),
        (f"Output Result directory: {output_path}", (10, 90), 0.40, 1),
    ]
    for text, pos, scale, thickness in header_lines:
        cv2.putText(header, text, pos, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    classes = list(counts.keys())
    num_classes = len(classes)
    fh = footer_height(num_classes)
    footer = np.ones((fh, w, 3), dtype=np.uint8) * 255

    for i, cls in enumerate(classes):
        row = i // SUMMARY_MAX_COLS
        col = i % SUMMARY_MAX_COLS
        cols_in_row = min(SUMMARY_MAX_COLS, num_classes - row * SUMMARY_MAX_COLS)
        col_w = w // cols_in_row

        x0 = col * col_w
        x1 = (col + 1) * col_w
        y0 = row * SUMMARY_FOOTER_ROW_H
        y1 = (row + 1) * SUMMARY_FOOTER_ROW_H
        mid = y0 + SUMMARY_FOOTER_ROW_H // 2

        cv2.rectangle(footer, (x0, y0), (x1, y1), (0, 0, 0), 1)

        lbl_sz, _ = cv2.getTextSize(cls, font, 0.6, 1)
        cv2.putText(footer, cls,
                    (x0 + (col_w - lbl_sz[0]) // 2, mid - 8),
                    font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        cnt_str = str(counts[cls])
        cnt_sz, _ = cv2.getTextSize(cnt_str, font, 0.8, 2)
        cv2.putText(footer, cnt_str,
                    (x0 + (col_w - cnt_sz[0]) // 2, y1 - 10),
                    font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return np.vstack([header, frame, footer])


def save_crop(
    frame: np.ndarray,
    box: np.ndarray,
    frame_w: int,
    frame_h: int,
    dest_dirs: list[Path],
    filename: str,
) -> None:
    x1, y1, x2, y2 = box
    cx1 = max(0, int(x1))
    cy1 = max(0, int(y1))
    cx2 = min(frame_w, int(x2))
    cy2 = min(frame_h, int(y2))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return
    for d in dest_dirs:
        cv2.imwrite(str(d / filename), crop)
