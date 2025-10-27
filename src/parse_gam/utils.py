from shapely.geometry import Polygon

import geopandas as gpd
import pandas as pd


def to_polygon(x):
    return Polygon(
        [
            (x.x_center - x.width / 2, x.y_center - x.height / 2),
            (x.x_center + x.width / 2, x.y_center - x.height / 2),
            (x.x_center + x.width / 2, x.y_center + x.height / 2),
            (x.x_center - x.width / 2, x.y_center + x.height / 2),
        ]
    )


def parse_yolo_predictions(predictions_path):
    df = pd.read_csv(
        predictions_path,
        names=["clas", "x_center", "y_center", "width", "height", "conf"],
        delimiter=" ",
    )

    df["geometry"] = df.apply(to_polygon, axis="columns")

    gdf = gpd.GeoDataFrame(df)

    return gdf


def project_onto_board(row):
    return {
        "clas": row["clas_pred"],
        "conf": row["conf_pred"],
        "board_index": row["board_index"],
        "x_center": (row.x_center_pred - row.x_center_board + row.width_board / 2)
        / row.width_board,
        "y_center": (row.y_center_pred - row.y_center_board + row.height_board / 2)
        / row.height_board,
        "width": row.width_pred / row.width_board,
        "height": row.height_pred / row.height_board,
    }


def iou(p1, p2):
    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter / union if union > 0 else 0


def deduplicate_gdf(gdf, iou_threshold: float = 0.8):
    """Deduplicate a geopandas frame based on IoU between geometries."""
    keep = []
    dropped = set()

    for i, geom_i in enumerate(gdf.geometry):
        if i in dropped:
            continue

        keep.append(i)

        for j in range(i + 1, len(gdf)):
            if j in dropped:
                continue
            geom_j = gdf.geometry[j]

            if iou(geom_i, geom_j) > iou_threshold:
                dropped.add(j)

    return gdf.loc[keep].reset_index(drop=True)
