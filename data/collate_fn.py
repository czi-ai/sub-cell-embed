import torch
import pandas as pd


def collate_fn_train(data):
    """
    data: is a list of tuples with (example, label, length)
          where 'example' is a tensor of arbitrary shape
          and label/length are scalars
    """
    x_i, x_j, protein_id, targets, mask = zip(*data)

    x_i = torch.cat(x_i, dim=0)
    x_j = torch.cat(x_j, dim=0)
    mask = torch.cat(mask, dim=0) if mask[0] is not None else None

    protein_id = torch.cat(protein_id, dim=0)
    targets = torch.cat(targets, dim=0)

    return x_i, x_j, protein_id, targets, mask


def collate_fn_test(data):
    """
    data: is a list of tuples with (example, label, length)
          where 'example' is a tensor of arbitrary shape
          and label/length are scalars
    """
    x_i, targets, antibody_metadata = zip(*data)

    x_i = torch.cat(x_i, dim=0)
    targets = torch.cat(targets, dim=0)
    metadata = pd.concat(antibody_metadata, axis=0).reset_index(drop=True)

    return x_i, targets, metadata


def collate_fn(data):
    """
    data: is a list of tuples with (example, label, length)
          where 'example' is a tensor of arbitrary shape
          and label/length are scalars
    """
    x_i, x_j, protein_id, protein_emb, targets, locations, metadata_tuple = zip(*data)
    # images, masks, protein_id, protein_emb, targets, locations = zip(*data)
    x_i = torch.cat(x_i, dim=0)
    x_j = torch.cat(x_j, dim=0) if x_j[0] is not None else None
    protein_id = torch.cat(protein_id, dim=0)

    protein_emb = (
        [item for sublist in protein_emb for item in sublist]
        if protein_emb[0] is not None
        else None
    )

    targets = torch.cat(targets, dim=0)
    locations = [item for sublist in locations for item in sublist]

    if metadata_tuple[0] is None:
        return x_i, x_j, protein_id, protein_emb, targets, locations, None
    else:
        metadata_df = pd.concat(metadata_tuple, axis=0).reset_index(drop=True)

    return x_i, x_j, protein_id, protein_emb, targets, locations, metadata_df
