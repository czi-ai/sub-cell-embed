import pandas as pd

df = pd.read_csv("/Users/it-user/Downloads/9606.protein.info.v12.0 (1).txt", sep="\t")
hpa_df = pd.read_csv("stats/IF-image-v23-filtered.csv")
hpa_genes = hpa_df["gene_names"].str.lower().unique().tolist()
df = df[df["preferred_name"].str.lower().isin(hpa_genes)]

links_df = pd.read_csv(
    "/Users/it-user/Downloads/9606.protein.links.v12.0 (1).txt", sep=" "
)
links_df = links_df[links_df["protein1"].isin(df["#string_protein_id"])]
links_df = links_df[links_df["protein2"].isin(df["#string_protein_id"])]

links_df["protein1"] = links_df["protein1"].map(
    df.set_index("#string_protein_id")["preferred_name"]
)
links_df["protein2"] = links_df["protein2"].map(
    df.set_index("#string_protein_id")["preferred_name"]
)
links_df.to_csv("stats/9606.protein.links.v12.0.filtered.txt", index=False)
