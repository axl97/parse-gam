from ultralytics import YOLO
import tqdm
from pathlib import Path
import argparse
import cv2


def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=1280)

    return parser.parse_args()


def main():
    args = __parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output.mkdir(exist_ok=True, parents=True)
    prediction_dir = args.output / "predictions"
    prediction_dir.mkdir(exist_ok=True, parents=True)
    frames_dir = args.output / "frames"
    frames_dir.mkdir(exist_ok=True, parents=True)

    success = True
    batch_counter = 0
    frame_counter = 0
    frame_buffer = []

    pbar = tqdm.tqdm(total=total_frames, desc="Predicting...", unit="frame")
    while success:
        success, frame = cap.read()

        frame_buffer.append(frame)
        batch_counter += 1

        if batch_counter >= args.batch_size or not success:
            result = model(
                frame_buffer,
                batch=True,
                verbose=False,
                imgsz=args.imgsz,
            )

            for r in result:
                frame_counter += 1
                r.save_txt(
                    prediction_dir / f"frame_{frame_counter}.txt", save_conf=True
                )
                r.save(frames_dir / f"frame_{frame_counter}.jpg", labels=False)
            pbar.update(len(frame_buffer))
            frame_buffer.clear()
            batch_counter = 0


if __name__ == "__main__":
    main()
