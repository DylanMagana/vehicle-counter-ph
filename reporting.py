import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VehicleRecord:
    filename: str
    processingtime: float
    label: str
    score: float


def write_report1(
    path: Path,
    bucket_counts: dict[int, int],
    total_secs: int,
    bucket_secs: int,
) -> None:
    buckets = range(0, total_secs + bucket_secs, bucket_secs)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "vehicles"])
        writer.writeheader()
        for b in buckets:
            writer.writerow({"time": b, "vehicles": bucket_counts.get(b, 0)})


def write_rpt_vehbytype(path: Path, records: list[VehicleRecord]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "processingtime", "label", "score"])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "filename": r.filename,
                "processingtime": r.processingtime,
                "label": r.label,
                "score": r.score,
            })
