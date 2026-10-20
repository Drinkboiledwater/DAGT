import pandas as pd
from sklearn.model_selection import train_test_split

# Load label file (columns: picture, mos)
csv_path = "/home/jackyang/WMJ/derain_datasets/mos1.csv"
df = pd.read_csv(csv_path)

# Extract scene id from filename, e.g. "0001_xxx.png" -> "0001"
df["scene"] = df["picture"].str.split("_").str[0]
assert df["scene"].notna().all(), "Filename does not match <id>_<rest>.png"

# Split scenes 7:1:2
scenes = df["scene"].unique()
print(f"Total samples: {len(df)}, unique scenes: {len(scenes)}")

# 70% train, 30% temp (val + test)
train_scenes, temp_scenes = train_test_split(
    scenes, test_size=0.3, random_state=39, shuffle=True
)
# Split temp into val / test at 1:2
val_scenes, test_scenes = train_test_split(
    temp_scenes, test_size=2 / 3, random_state=39, shuffle=True
)

train_df = df[df["scene"].isin(train_scenes)].reset_index(drop=True)
val_df = df[df["scene"].isin(val_scenes)].reset_index(drop=True)
test_df = df[df["scene"].isin(test_scenes)].reset_index(drop=True)


def report(name, d):
    print(f"[{name}] scenes: {d['scene'].nunique():3d}, "
          f"samples: {len(d):5d}, ratio: {len(d) / len(df) * 100:5.2f}%")


report("train", train_df)
report("val", val_df)
report("test", test_df)

# Verify no scene overlap
s_tr, s_va, s_te = set(train_scenes), set(val_scenes), set(test_scenes)
assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), \
    "Scene overlap detected"
print("Scene split: no overlap")

# Save splits
train_df.drop(columns=["scene"]).to_csv("train.csv", index=False)
val_df.drop(columns=["scene"]).to_csv("val.csv", index=False)
test_df.drop(columns=["scene"]).to_csv("test.csv", index=False)
