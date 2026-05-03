from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from barcodeplot.cli import main


def _write_dummy_frames(frame_dir: Path, count: int) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, count + 1):
        image = np.zeros((48, 80, 3), dtype=np.uint8)
        image[:, :, 2] = idx * 20
        cv2.imwrite(str(frame_dir / f"{idx:06d}.png"), image)


def test_cli_plot(tmp_path: Path):
    gt_path = tmp_path / "gt.csv"
    pred_path = tmp_path / "pred.csv"
    gt_path.write_text("frame_number,gt_binary\n1,0\n2,1\n3,0\n", encoding="utf-8")
    pred_path.write_text(
        "frame_number,contact_label\n1,No Contact\n2,Portable Object\n3,No Contact\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "barcode.png"
    code = main(
        [
            "plot",
            "--track",
            f"{gt_path}:GT",
            "--track",
            f"{pred_path}:Pred",
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    assert out_path.exists()


def test_cli_video(tmp_path: Path):
    frame_dir = tmp_path / "frames"
    _write_dummy_frames(frame_dir, 3)
    gt_path = tmp_path / "gt.csv"
    pred_path = tmp_path / "pred.csv"
    gt_path.write_text("frame_number,gt_binary\n1,0\n2,1\n3,0\n", encoding="utf-8")
    pred_path.write_text(
        "frame_number,value\n1,0\n2,1\n3,1\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "timeline.mp4"
    code = main(
        [
            "video",
            "--frame-dir",
            str(frame_dir),
            "--track",
            f"{gt_path}:GT",
            "--track",
            f"{pred_path}:Pred",
            "--fps",
            "10",
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    assert out_path.exists()
