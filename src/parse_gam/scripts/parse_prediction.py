from pathlib import Path
import json

import multiprocessing
import geopandas as gpd
import pandas as pd
import argparse
from tqdm import tqdm
from parse_gam.utils import (
    deduplicate_gdf,
    to_polygon,
    project_onto_board,
    parse_yolo_predictions,
)

CLASS_MAPPING = {
    "BOARD": 0,
    "CHECKER_P1": 1,
    "CHECKER_P2": 2,
    "DIE": 3,
    "HAND": 4,
    "POINT": 5,
}


def __parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("predictions", type=Path)
    args.add_argument("output", type=Path)

    args.add_argument("--num-proc", type=int, default=4)

    return args.parse_args()


def parse_half_board_state(gdf):
    """
    Parse a half-board's state.

    Assumes x and y center columns are coordinates within [0,1] of the center of the checker position.
    Divides the board into 6 Backgammon 'Points' on the upper and lower sides. Assumes the points cover the length of [0,1]
    """
    state = {}

    CHECKER_POINT_X_TOLERANCE = 0.07
    CHECKER_P1_CLASS = CLASS_MAPPING["CHECKER_P1"]
    CHECKER_P2_CLASS = CLASS_MAPPING["CHECKER_P2"]
    DIE_CLASS = CLASS_MAPPING["DIE"]

    for half in ["UPPER", "LOWER"]:
        for h_point_index in range(0, 6):
            h_pos = h_point_index / 6 + (1 / 6 / 2)
            if half == "LOWER":
                # Lower half
                d = gdf[
                    (gdf.y_center > 0.5)
                    & ((gdf.x_center - h_pos).abs() < CHECKER_POINT_X_TOLERANCE)
                    & (gdf.clas.isin([CHECKER_P1_CLASS, CHECKER_P2_CLASS]))
                ].reset_index(drop=True)
                point_index = 6 - h_point_index
            elif half == "UPPER":
                d = gdf[
                    (gdf.y_center < 0.5)
                    & ((gdf.x_center - h_pos).abs() < CHECKER_POINT_X_TOLERANCE)
                    & (gdf.clas.isin([CHECKER_P1_CLASS, CHECKER_P2_CLASS]))
                ].reset_index(drop=True)
                point_index = 7 + h_point_index
            else:
                raise RuntimeError()

            class_counts = d.groupby("clas").size().reset_index(name="count")

            # Find the class with the most lines
            top_classes = class_counts.nlargest(1, "count", keep="all")

            # Determine the selected class
            if top_classes.empty:
                # No classes found, default to CHECKER_P2
                selected_class_index = CLASS_MAPPING["CHECKER_P2"]
            elif len(top_classes) > 1 and top_classes["count"].nunique() == 1:
                # Multiple classes tied for the highest count, use mean confidence to break tie
                mean_confidence = d.groupby("clas")["conf"].mean()
                selected_class = mean_confidence.idxmax()
                selected_class_index = selected_class
            else:
                # Single class with the highest count
                selected_class_index = int(top_classes["clas"].iloc[0])

            num_checkers = deduplicate_gdf(d).shape[0]
            val = (
                num_checkers
                if selected_class_index == CLASS_MAPPING["CHECKER_P2"]
                else -num_checkers
            )

            state[f"Point_{point_index}"] = val

    # Detect if dice are on the board
    num_dice = deduplicate_gdf(gdf[gdf.clas == DIE_CLASS], iou_threshold=0.4).shape[0]

    state["dice"] = num_dice

    return state


def parse_board_state(predictions: gpd.GeoDataFrame) -> dict | None:
    BOARD_CLASS = CLASS_MAPPING["BOARD"]
    CHECKER_P1_CLASS = CLASS_MAPPING["CHECKER_P1"]
    CHECKER_P2_CLASS = CLASS_MAPPING["CHECKER_P2"]
    HAND_CLASS = CLASS_MAPPING["HAND"]
    DIE_CLASS = CLASS_MAPPING["DIE"]

    # There are two 'boards' give the one to the left index 0 and the one to the right index 1
    boards = (
        predictions[predictions.clas == BOARD_CLASS]
        .sort_values(by="x_center")
        .reset_index(drop=True)
    )
    boards.index.names = ["board_index"]
    boards = boards.reset_index()

    boards = deduplicate_gdf(boards)

    if boards.shape[0] != 2:
        # print("Not two board predictions")
        return {"status": "UNPARSEABLE"}

    # Project each prediction on a [0,1] coordinate system within its respective board
    projected = boards.sjoin(
        predictions, how="inner", lsuffix="board", rsuffix="pred"
    ).apply(project_onto_board, axis="columns", result_type="expand")

    projected = projected[
        projected.clas.isin([CHECKER_P1_CLASS, CHECKER_P2_CLASS, HAND_CLASS, DIE_CLASS])
    ]

    if projected.shape[0] == 0:
        # print("No non-board predictions")
        return {"STATUS": "unparseable"}

    # Drop all predictions expcept ofr the Checkers P1 and Checkers P2 classes
    projected["geometry"] = projected.apply(to_polygon, axis="columns")

    projected = gpd.GeoDataFrame(projected)

    # Check if there is a hand within any of the boards. If so say return a status OBSCURED
    if (projected.clas == HAND_CLASS).sum() > 0:
        # print("Hand detected")
        return {"status": "OBSCURED"}

    # Now we have all Checker predictions with [0,1] x and y coordinates within their respective board along with board index
    # Count checker positions in each board half separately and then merge the board-half states
    state_board_1 = parse_half_board_state(projected[projected["board_index"] == 0])
    state_board_2 = parse_half_board_state(projected[projected["board_index"] == 1])

    full_state = {}
    # merge two states
    for x in range(1, 25):
        if x > 0 and x <= 6:
            full_state[f"Point_{x}"] = state_board_2[f"Point_{x}"]
        elif x > 6 and x <= 12:
            full_state[f"Point_{x}"] = state_board_1[f"Point_{x - 6}"]
        elif x > 12 and x <= 18:
            full_state[f"Point_{x}"] = state_board_1[f"Point_{x - 6}"]
        elif x > 18 and x <= 24:
            full_state[f"Point_{x}"] = state_board_2[f"Point_{x - 12}"]

    full_state["board_1_dice"] = state_board_1["dice"]
    full_state["board_2_dice"] = state_board_2["dice"]

    full_state["status"] = "VALID"

    return full_state


def parse_single_prediction(prediction_path, output_path):
    predictions = parse_yolo_predictions(prediction_path)

    parse_output = parse_board_state(predictions)

    if parse_output is None:
        # print("Unable to parse board state")
        board_state = {"status": "UNPARSEABLE"}
    else:
        board_state = parse_output

    # Add file index
    try:
        file_index = int(Path(prediction_path).stem.split("_")[-1])
    except Exception as _:
        file_index = None

    board_state["file_index"] = file_index

    with output_path.open("w") as io:
        json.dump(board_state, io, indent=4)


def _process_file(args):
    pred, output_dir = args
    output_name = output_dir / pred.with_suffix(".json").name
    parse_single_prediction(pred, output_name)
    return pred


def main():
    args = __parse_args()

    if args.predictions.is_dir():
        args.output.mkdir(exist_ok=True)
        all_preds = list(args.predictions.iterdir())
        if args.num_proc > 1:
            with multiprocessing.Pool(processes=args.num_proc) as pool:
                list(
                    tqdm(
                        pool.imap_unordered(
                            _process_file,
                            [(pred, args.output) for pred in all_preds],
                            chunksize=max(1, len(all_preds) // (args.num_proc * 20)),
                        ),
                        total=len(all_preds),
                        desc="Parsing predictions",
                    )
                )
        else:
            for pred in tqdm(all_preds, desc="Parsing predictions"):
                _process_file((pred, args.output))

    else:
        parse_single_prediction(args.predictions, args.output)


if __name__ == "__main__":
    main()
