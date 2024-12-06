import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torchvision.utils import make_grid
import torch
from umap import UMAP
import seaborn as sns
import colorcet as cc
from sklearn.decomposition import NMF, PCA

LOCATION_MAP = pd.read_csv("stats/location_group_mapping.csv")


def get_display_image(input, color_channels):
    color_ip = []
    for i, color in enumerate(color_channels):
        lut = np.array(
            pd.read_csv(f"utils/colormaps/{color}.lut", sep="\t", index_col="Index")
        )[None, ...].astype(np.uint8)
        ch_img = input[..., i][..., None].repeat(3, axis=-1)
        image_lut = cv2.LUT(ch_img, lut)
        color_ip.append(image_lut)
    color_ip = np.sum(color_ip, axis=0).clip(0, 255).astype(np.uint8)
    return color_ip


def get_locations(one_hot, grouping_category=2):
    unique_cats = LOCATION_MAP["Original annotation"].unique().tolist()
    grouping_map = {
        row["Original annotation"]: row[f"Grouping {grouping_category}"]
        for _, row in LOCATION_MAP.iterrows()
    }
    locations = []
    for loc in one_hot:
        if np.sum(loc) > 1:
            locations.append("Multilocalizing")
        else:
            locations.append(unique_cats[np.argmax(loc)])
    return locations


def get_feat_nmf(input, feat, color_channels):
    input, feat = input[:4], feat[:4]

    n_rows = input.shape[0] // 4
    n_cols = 4

    combined_grid_list = []
    for i in range(input.shape[0]):
        inputrow = (input[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
            np.uint8
        )
        inputrow = get_display_image(inputrow, color_channels)

        featrow = feat[i].permute(1, 2, 0).detach().cpu().numpy()
        w, h, c = featrow.shape
        featrow = featrow.reshape(w * h, c)
        featrow = PCA(n_components=3).fit_transform(featrow)
        featrow = featrow.reshape(w, h, 3)
        for j in range(3):
            featrow[..., j] = (featrow[..., j] - featrow[..., j].min()) / (
                featrow[..., j].max() - featrow[..., j].min()
            )
        featrow = (featrow * 255).astype(np.uint8)
        featrow = cv2.resize(
            featrow,
            (inputrow.shape[1], inputrow.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        combined_grid_list.append(np.concatenate([inputrow, featrow], axis=1))

    combined_grid = []
    for i in range(n_rows):
        combined_grid.append(
            np.concatenate(combined_grid_list[i * n_cols : (i + 1) * n_cols], axis=0)
        )
    combined_grid = np.concatenate(combined_grid, axis=1)
    return combined_grid


def get_attn(input, attn, color_channels):
    input, recon, attn = input[:4], recon[:4], attn[:4]

    combined_grid = []
    for i in range(input.shape[0]):
        inputrow = (input[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
            np.uint8
        )
        inputrow = get_display_image(inputrow, color_channels)

        attnrow = make_grid(
            attn[i].unsqueeze(1).repeat(1, 3, 1, 1),
            # * torch.tensor(inputrow).permute(2, 0, 1).unsqueeze(0).to(input.device),
            normalize=True,
            # scale_each=True,
            nrow=attn[i].shape[0],
            padding=0,
        )
        attnrow = (
            (attnrow.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
        )
        attnrow = 255 - attnrow

        combined_grid.append(np.concatenate([inputrow, attnrow], axis=1))

    combined_grid = np.concatenate(combined_grid, axis=0)
    return combined_grid


def save_feat_nmf(
    input, feat, epoch, batch_idx, color_channels, save_folder, tag="feat"
):
    input, feat = input[:4], feat[:4]

    n_rows = input.shape[0] // 4
    n_cols = 4

    combined_grid_list = []
    for i in range(input.shape[0]):
        inputrow = (input[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
            np.uint8
        )
        inputrow = get_display_image(inputrow, color_channels)

        featrow = feat[i].permute(1, 2, 0).detach().cpu().numpy()
        w, h, c = featrow.shape
        featrow = featrow.reshape(w * h, c)
        featrow = PCA(n_components=3).fit_transform(featrow)
        featrow = featrow.reshape(w, h, 3)
        for j in range(3):
            featrow[..., j] = (featrow[..., j] - featrow[..., j].min()) / (
                featrow[..., j].max() - featrow[..., j].min()
            )
        featrow = (featrow * 255).astype(np.uint8)
        featrow = cv2.resize(
            featrow,
            (inputrow.shape[1], inputrow.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        combined_grid_list.append(np.concatenate([inputrow, featrow], axis=1))

    combined_grid = []
    for i in range(n_rows):
        combined_grid.append(
            np.concatenate(combined_grid_list[i * n_cols : (i + 1) * n_cols], axis=0)
        )
    combined_grid = np.concatenate(combined_grid, axis=1)

    cv2.imwrite(
        f"{save_folder}/epoch_{epoch}_batch_{batch_idx}_{tag}.png",
        cv2.cvtColor(combined_grid, cv2.COLOR_RGB2BGR),
    )


def plot_feature_umaps(
    features_dict, feat_types, epoch, save_folder, grouping_category=2
):
    one_hot = features_dict["targets"]
    locations = get_locations(one_hot, grouping_category=grouping_category)

    print("Plotting UMAPs", flush=True)
    for feat_type in feat_types:
        umap_features = UMAP(
            n_jobs=-1,
            n_components=2,
            metric="euclidean",
        ).fit_transform(features_dict[feat_type])
        df = pd.DataFrame(
            {
                "x": umap_features[:, 0],
                "y": umap_features[:, 1],
                "location": locations,
            }
        )

        ax = sns.scatterplot(
            x="x",
            y="y",
            hue="location",
            data=df,
            legend="brief",
            palette=sns.color_palette(cc.glasbey_dark, len(df["location"].unique())),
            s=2,
            alpha=0.5,
        )
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=(1.02, 1),
            loc=2,
            borderaxespad=0.0,
            markerscale=4,
        )
        save_name = f"umap_location_{feat_type}_epoch_{epoch}.png"
        plt.savefig(
            f"{save_folder}/{save_name}",
            dpi=300,
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.close()


def save_grid_images(image, epoch, batch_idx, color_channels, save_folder, tag="input"):
    image_grid = make_grid(image, normalize=True, value_range=(0, 1), scale_each=True)
    image_grid = image_grid.permute(1, 2, 0).detach().cpu().numpy()
    image_grid = (image_grid * 255).astype(np.uint8)
    if tag != "attn":
        color_img = []
        for i, color in enumerate(color_channels):
            lut = np.array(
                pd.read_csv(f"utils/colormaps/{color}.lut", sep="\t", index_col="Index")
            )[None, ...].astype(np.uint8)
            image_lut = cv2.LUT(image_grid[..., i][..., None].repeat(3, axis=-1), lut)
            color_img.append(image_lut)
        color_img = np.sum(color_img, axis=0).clip(0, 255).astype(np.uint8)
    else:
        if image_grid.shape[1] == 1:
            colormap = plt.get_cmap("inferno")
            color_img = (colormap(image_grid[..., 0])[..., :3] * 255).astype(np.uint8)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
        else:
            color_img = (image_grid * 255).astype(np.uint8)
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(
        f"{save_folder}/epoch_{epoch}_batch_{batch_idx}_{tag}.png",
        cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR),
    )


def save_overlay_attn(
    input, recon, attn, epoch, batch_idx, color_channels, save_folder, tag="ipattn"
):
    input, recon, attn = input[:8], recon[:8], attn[:8]

    combined_grid = []
    for i in range(input.shape[0]):
        inputrow = (input[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
            np.uint8
        )
        inputrow = get_display_image(inputrow, color_channels)
        reconrow = (
            recon[i, -1, :, :].unsqueeze(-1).repeat(1, 1, 3).detach().cpu().numpy()
            * 255
        ).astype(np.uint8)
        reconrow[:, :, [0, 2]] = 0
        # reconrow = get_display_image(reconrow, color_channels)

        attnrow = make_grid(
            attn[i].unsqueeze(1).repeat(1, 3, 1, 1),
            # * torch.tensor(inputrow).permute(2, 0, 1).unsqueeze(0).to(input.device),
            normalize=True,
            # scale_each=True,
            nrow=attn[i].shape[0],
            padding=0,
        )
        attnrow = (
            (attnrow.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
        )
        attnrow = 255 - attnrow

        combined_grid.append(np.concatenate([inputrow, reconrow, attnrow], axis=1))  #

    combined_grid = np.concatenate(combined_grid, axis=0)
    cv2.imwrite(
        f"{save_folder}/epoch_{epoch}_batch_{batch_idx}_{tag}.png",
        cv2.cvtColor(combined_grid, cv2.COLOR_RGB2BGR),
    )


def save_feat_nmf(
    input, feat, epoch, batch_idx, color_channels, save_folder, tag="feat"
):
    input, feat = input[:8], feat[:8]

    n_rows = input.shape[0] // 4
    n_cols = 4

    combined_grid_list = []
    for i in range(input.shape[0]):
        inputrow = (input[i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
            np.uint8
        )
        inputrow = get_display_image(inputrow, color_channels)

        featrow = feat[i].permute(1, 2, 0).detach().cpu().numpy()
        w, h, c = featrow.shape
        featrow = featrow.reshape(w * h, c)
        featrow = PCA(n_components=3).fit_transform(featrow)
        featrow = featrow.reshape(w, h, 3)
        for j in range(3):
            featrow[..., j] = (featrow[..., j] - featrow[..., j].min()) / (
                featrow[..., j].max() - featrow[..., j].min()
            )
        featrow = (featrow * 255).astype(np.uint8)
        featrow = cv2.resize(
            featrow,
            (inputrow.shape[1], inputrow.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        combined_grid_list.append(np.concatenate([inputrow, featrow], axis=1))

    combined_grid = []
    for i in range(n_rows):
        combined_grid.append(
            np.concatenate(combined_grid_list[i * n_cols : (i + 1) * n_cols], axis=0)
        )
    combined_grid = np.concatenate(combined_grid, axis=1)

    cv2.imwrite(
        f"{save_folder}/epoch_{epoch}_batch_{batch_idx}_{tag}.png",
        cv2.cvtColor(combined_grid, cv2.COLOR_RGB2BGR),
    )
