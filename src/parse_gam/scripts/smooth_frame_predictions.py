import json
import numpy as np
import re
import pandas as pd
from pathlib import Path
import argparse


def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("states", type=Path)
    parser.add_argument("output", type=Path)

    return parser.parse_args()


def load_states(path):
    states = []
    for p in path.iterdir():
        with p.open() as io:
            s = json.load(io)
            s["filename"] = p.name
            states.append(s)

    return pd.DataFrame(states)


def extract_index(filename):
    match = re.search(r"_(\d+)\.json$", filename)
    if match:
        return int(match.group(1))
    return np.nan


def smooth_states(df, imputation_method="linear", window_size=9):
    point_cols = [x for x in df.columns if x.startswith("Point_")]

    dice_cols = ["board_1_dice", "board_2_dice"]

    smooth_cols = [*point_cols, *dice_cols]

    smoothed_df = df.copy()

    smoothed_df["file_index"] = smoothed_df.filename.apply(extract_index)

    smoothed_df = smoothed_df.sort_values(["file_index"]).reset_index(drop=True)

    # Drop frames that are invalid
    smoothed_df = smoothed_df[smoothed_df.status == "VALID"]

    for col in smooth_cols:
        if imputation_method == "linear":
            smoothed_df[col] = smoothed_df[col].interpolate(
                method="linear", limit_direction="both"
            )
        elif imputation_method == "ffill":
            smoothed_df[col] = smoothed_df[col].fillna(method="ffill")
        elif imputation_method == "bfill":
            smoothed_df[col] = smoothed_df[col].fillna(method="bfill")
        else:
            raise ValueError("imputation_method must be 'linear', 'ffill', or 'bfill'")

    for col in point_cols:
        smoothed_df[col] = (
            smoothed_df[col]
            .rolling(window=window_size, min_periods=1, center=True)
            .mean()
            .round()
            .astype("Int64")
        )
    return smoothed_df


def save_states(df, path):
    path.mkdir(exist_ok=True)

    for r in df.iloc:
        data = r.to_dict()
        filename = data.pop("filename")

        with (path / filename).open("w") as io:
            json.dump(data, io)


def main():
    args = __parse_args()

    df = load_states(args.states)

    smoothed_df = smooth_states(df)

    save_states(smoothed_df, args.output)


if __name__ == "__main__":
    main()
