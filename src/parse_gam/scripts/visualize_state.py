import argparse
from PIL import Image
import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.patches as patches
import json
from pathlib import Path


def __parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("states", type=Path)
    args.add_argument("--frames", type=Path, default=None)

    args.add_argument("output", type=Path)

    return args.parse_args()


def visualize_state(state, show=False):
    f, ax = plt.subplots(1, 1, figsize=(16, 8))

    left_board = patches.Rectangle(
        (-4.0, -2.5), 3.5, 5, linewidth=1, edgecolor="black", facecolor="none"
    )
    right_board = patches.Rectangle(
        (0.5, -2.5), 3.5, 5, linewidth=1, edgecolor="black", facecolor="none"
    )
    ax.add_patch(left_board)
    ax.add_patch(right_board)

    point_x = {
        **{f"Point_{i}": 3.5 - (i - 1) * 0.5 for i in range(1, 7)},
        **{f"Point_{i}": -1.0 - (i - 7) * 0.5 for i in range(7, 13)},
        **{f"Point_{i}": -3.5 + (i - 13) * 0.5 for i in range(13, 19)},
        **{f"Point_{i}": 1.0 + (i - 19) * 0.5 for i in range(19, 25)},
    }

    # REALLY UGLY FIX THIS LATER
    status = state.pop("status", "VALID")

    if status == "VALID":
        checker_radius = 0.2
        max_checkers = 6
        for point, count in state.items():
            if point not in point_x:
                continue
            x = point_x[point]
            is_top = int(point.split("_")[1]) in range(13, 25)
            y_start = 2.5 if is_top else -2.5
            y_end = 1.5 if is_top else -1.5
            # Draw point line (triangle approximated as a line)
            plt.plot([x, x], [y_start, y_end], color="black", linewidth=1)

            abs_count = abs(count)
            if abs_count > 0:
                color = "white" if count < 0 else "black"
                for i in range(min(abs_count, max_checkers)):
                    y = y_start + (-1 if is_top else 1) * (
                        i * 2 * checker_radius + checker_radius
                    )
                    circle = patches.Circle(
                        (x, y), checker_radius, facecolor=color, edgecolor="black"
                    )
                    ax.add_patch(circle)
                # If more than max_checkers, add a text label
                if abs_count > max_checkers:
                    y = y_end + (-1 if is_top else 1) * (
                        max_checkers * 2 * checker_radius + checker_radius
                    )

                    plt.text(x, y, str(abs_count), ha="center", va="center", fontsize=8)
    elif status in ["UNPARSEABLE", "OBSCURED"]:
        ax.text(
            x=-1.0,
            y=0,
            s="UNABLE TO PARSE BOARD",
            fontsize=40,
            color="red",
            ha="center",
            va="center",
            zorder=10,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
        )

    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.axis("off")

    if show:
        plt.show()
        return f, ax
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(f)

        buf.seek(0)

        return buf


def join_state_and_frame_visualizations(state_vis, frame_path, output_path):
    state_img = Image.open(state_vis)
    frame_img = Image.open(frame_path)

    state_width, state_height = state_img.size
    frame_width, frame_height = frame_img.size
    # new_height = min(state_height, frame_height)
    new_height = frame_height
    state_img = state_img.resize(
        (int(state_height * new_height / state_height), new_height)
    )
    frame_img = frame_img.resize(
        (int(frame_height * new_height / frame_height), new_height)
    )

    # Create a new blank image to hold both
    total_width = state_img.width + frame_img.width
    combined_img = Image.new("RGB", (total_width, new_height))
    combined_img.paste(frame_img, (0, 0))
    combined_img.paste(state_img, (frame_img.width, 0))

    # Save the combined image
    combined_img.save(output_path)


def visualize_single_state(state_path, frame_path, output_path):
    with state_path.open() as io:
        state = json.load(io)

    if frame_path is not None and not frame_path.exists():
        raise ValueError("No such frame file exists")

    state_vis = visualize_state(state)

    if frame_path is None:
        # Just save the state visualization
        with output_path.open("w") as io:
            io.write(state_vis.read())
    else:
        # visualize the state and the frame capture side by side
        join_state_and_frame_visualizations(state_vis, frame_path, output_path)


def main():
    args = __parse_args()

    if args.states.is_dir():
        if not args.frames.is_dir():
            raise ValueError("State is directory but frames is not.")

        args.output.mkdir(exist_ok=True)

        for state_path in tqdm(args.states.iterdir()):
            frame_path = args.frames / state_path.with_suffix(".jpg").name

            frame_index = int(state_path.stem.split("_")[-1])
            output_path = args.output / f"vis_{frame_index:04d}.jpg"

            visualize_single_state(state_path, frame_path, output_path)
    else:
        visualize_single_state(state_path, frame_path, output_path)


if __name__ == "__main__":
    main()
