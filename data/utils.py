import random

import numpy as np
import pandas as pd
from sklearn import preprocessing

LOCATION_MAP = pd.read_csv("stats/location_group_mapping.csv")


def standardize(im):
    min = np.min(im, axis=(1, 2), keepdims=True)
    max = np.max(im, axis=(1, 2), keepdims=True)

    im = (im - min) / (max - min + 1e-8)
    im = np.clip(im, 0, 1)
    return im.astype(np.float32)


def min_max_standardize(im, min_perc=0, max_perc=100):
    min_val = np.percentile(im, min_perc, axis=(1, 2), keepdims=True)
    max_val = np.percentile(im, max_perc, axis=(1, 2), keepdims=True)

    im = (im - min_val) / (max_val - min_val + 1e-8)
    im = np.clip(im, 0, 1)
    return im.astype(np.float32)


def normalization(im, dmso_mean=None, dmso_std=None):
    dmso_mean = (
        dmso_mean if dmso_mean is not None else im.mean(axis=(1, 2), keepdims=True)
    )
    dmso_std = dmso_std if dmso_std is not None else im.std(axis=(1, 2), keepdims=True)
    im = (im - dmso_mean) / dmso_std
    return im.astype(np.float32)


def min_max_normalization(im, min_perc=1, max_perc=99):
    min_val = np.percentile(im, min_perc, axis=(1, 2), keepdims=True)
    max_val = np.percentile(im, max_perc, axis=(1, 2), keepdims=True)

    im = (im - min_val) / (max_val - min_val + 1e-8)
    im = np.clip(im, 0, 1)
    mean, std = im.mean(), im.std()
    im = (im - mean) / std
    return im.astype(np.float32)


def one_hot_encode_locations(df):
    locations = df["locations"].str.split(",").tolist()
    unique_cats = LOCATION_MAP["Original annotation"].unique().tolist()
    one_hot = [
        [1 if loc in location else 0 for loc in unique_cats] for location in locations
    ]
    df[unique_cats] = pd.DataFrame(one_hot, index=df.index)
    return df, unique_cats


def preprocess_locations(df, grouping_category):
    df["locations"] = df["locations"].str.split(",")
    grouping_map = {
        row["Original annotation"]: row[f"Grouping {grouping_category}"]
        for _, row in LOCATION_MAP.iterrows()
    }
    df["locations"] = df["locations"].apply(lambda x: [grouping_map[loc] for loc in x])
    df["locations"] = df["locations"].apply(
        lambda x: "Multilocalizing" if len(x) > 1 else x[0]
    )
    labeler = preprocessing.LabelEncoder().fit(df["locations"].values)
    df["location2idx"] = labeler.transform(df["locations"].values)
    return df, labeler
