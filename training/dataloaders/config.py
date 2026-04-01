import os

# ---------------------------------------------------------------------------
# Data root configuration
# ---------------------------------------------------------------------------
# Set the DAGE_DATA_ROOT environment variable to point to your local data
# directory, or modify the default path below. All dataset files (parquets,
# images, depth maps, etc.) are expected to live under this root.
#
# Expected directory layout:
#   DATA_ROOT/
#   ├── parquets/          # Parquet index files for each dataset
#   ├── waymo/             # Raw/processed Waymo frames
#   ├── tartanair/         # Raw/processed TartanAir frames
#   ├── ...                # Other datasets
# ---------------------------------------------------------------------------

DATA_ROOT = "<PATH TO TRAINING DATA>"

def _data_path(*parts):
    """Construct a path under DATA_ROOT."""
    return os.path.join(DATA_ROOT, *parts)


DATASET_TASK = {
    "depth": {
        "hypersim": 59543,
        "virtual_kitti_2": 21260,
        "tartanair": 116984,
        "point_odyssey": 239022,
        "spring": 5000,
        "arkitscenes": 0,
        "midair": 50000,
        "scannetpp": 0,
        "mvs_synth": 0,
        "co3d": 0,
        "bedlam": 0,
        "blendedmvs": 0,
        "megadepth": 0,
        "gta_sfm": 0,
        "mp3d": 0,
    },
    "normals": {
        "hypersim": 59543,
        "virtual_kitti_2": 21260,
        "point_odyssey": 239022,
    },
    "pointmap": {
        "hypersim": 59543,
        "virtual_kitti_2": 21260,
        "tartanair": 116984,
        "point_odyssey": 239022,
        "spring": 5000,
        "arkitscenes": 0,
        "midair": 50000,
        "scannetpp": 0,
        "mvs_synth": 0,
        "co3d": 0,
        "bedlam": 0,
        "blendedmvs": 0,
        "megadepth": 0,
        "gta_sfm": 0,
        "mp3d": 0,
    },
    "prtmat": {
        "p3m_10k": 9421,
    },
}

DATASET_CONFIG = {
    "hypersim": {
        "prefix": _data_path("hypersim", "processed"),
        "parquet_path": _data_path("parquets", "hypersim_train.parquet"),
        "tasks": ["depth", "normals", "pointmap"],
        "is_video": False,
        "num_images": 59543,
    },
    "virtual_kitti_2": {
        "prefix": _data_path("virtual_kitti_2"),
        "parquet_path": _data_path("parquets", "virtual_kitti_2_train.parquet"),
        "tasks": ["depth", "normals", "pointmap"],
        "is_video": True,
        "num_images": 21260,
    },
    "p3m_10k": {
        "prefix": _data_path("P3M-10k"),
        "parquet_path": _data_path("parquets", "p3m_10k_train.parquet"),
        "tasks": ["prtmat"],
        "is_video": False,
        "num_images": 9421,
    },
    "tartanair": {
        "prefix": _data_path("tartanair"),
        "parquet_path": _data_path("parquets", "tartanair_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 116984,
    },
    "point_odyssey": {
        "prefix": _data_path("point_odyssey"),
        "parquet_path": _data_path("parquets", "point_odyssey_train.parquet"),
        "tasks": ["depth", "normals", "pointmap"],
        "is_video": True,
        "num_images": 239022,
    },
    "spring": {
        "prefix": _data_path("spring"),
        "parquet_path": _data_path("parquets", "spring_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 5000,
    },
    "dynamic_replica": {
        "prefix": _data_path("dynamic_replica"),
        "parquet_path": _data_path("parquets", "dynamic_replica_train.parquet"),
        "tasks": ["depth"],
        "is_video": True,
        "num_images": 150900,
    },
    "waymo": {
        "prefix": _data_path("waymo"),
        "parquet_path": _data_path("parquets", "waymo_train.parquet"),
        "tasks": ["depth"],
        "is_video": True,
        "num_images": 632324,
    },
    "arkitscenes": {
        "prefix": _data_path("arkitscenes"),
        "parquet_path": _data_path("parquets", "arkitscenes_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 943650,
    },
    "midair": {
        "prefix": _data_path("midair"),
        "parquet_path": _data_path("parquets", "midair_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 71535,
    },
    "virtual_kitti_2_video": {
        "prefix": _data_path("virtual_kitti_2"),
        "parquet_path": _data_path("parquets", "virtual_kitti_2_video_train.parquet"),
        "tasks": ["depth", "normals", "pointmap"],
        "is_video": True,
        "num_images": 21260,
    },
    "kubric": {
        "prefix": _data_path("kubric_processed_mix_3d"),
        "parquet_path": _data_path("parquets", "kubric_train.parquet"),
        "tasks": ["depth", "normals", "pointmap"],
        "is_video": True,
        "num_images": 135144,
    },
    "wildrgbd": {
        "prefix": _data_path("wildrgbd"),
        "parquet_path": _data_path("parquets", "wildrgbd_train.parquet"),
        "tasks": ["depth"],
        "is_video": True,
        "num_images": 144000,
    },
    "scannetpp": {
        "prefix": _data_path("scannetpp"),
        "parquet_path": _data_path("parquets", "scannetpp_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 32244,
    },
    "megadepth": {
        "prefix": _data_path("megadepth"),
        "parquet_path": _data_path("parquets", "megadepth_train.parquet"),
        "precomputed_sets_path": _data_path("megadepth", "megadepth_sets_64.npz"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 81000,
    },
    "matrixcity": {
        "prefix": _data_path("matrixcity"),
        "parquet_path": _data_path("parquets", "matrixcity_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 95242,
    },
    "gta_sfm": {
        "prefix": _data_path("gta_sfm"),
        "parquet_path": _data_path("parquets", "gta_sfm_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 17649,
    },
    "mvs_synth": {
        "prefix": _data_path("mvs_synth"),
        "parquet_path": _data_path("parquets", "mvs_synth_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 12000,
    },
    "mvs_synth_hr": {
        "prefix": _data_path("mvs_synth_hr"),
        "parquet_path": _data_path("parquets", "mvs_synth_hr_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 12000,
    },
    "co3d": {
        "prefix": _data_path("co3d"),
        "parquet_path": _data_path("parquets", "co3d_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 251745,
    },
    "bedlam": {
        "prefix": _data_path("bedlam"),
        "parquet_path": _data_path("parquets", "bedlam_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 13463,
    },
    "hypersim_video": {
        "prefix": _data_path("hypersim_video"),
        "parquet_path": _data_path("parquets", "hypersim_video_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 43665,
    },
    "blendedmvs": {
        "prefix": _data_path("blendedmvs"),
        "parquet_path": _data_path("parquets", "blendedmvs_train.parquet"),
        "overlap_path": _data_path("blendedmvs", "new_overlap.h5"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 115142,
    },
    "scannet": {
        "prefix": _data_path("scannet"),
        "parquet_path": _data_path("parquets", "scannet_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 233496,
    },
    "spatialvidhq_videoonly": {
        "prefix": _data_path("spatialvidhq_videoonly"),
        "parquet_path": _data_path("parquets", "spatialvidhq_videoonly_1_74.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 122830,
    },
    "dl3dv": {
        "prefix": _data_path("dl3dv"),
        "parquet_path": _data_path("parquets", "dl3dv_1k_8k.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 0,
    },
    "stereo4d": {
        "prefix": _data_path("stereo4d"),
        "parquet_path": _data_path("parquets", "stereo4d.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 47477,
    },
    "habitat3d": {
        "prefix": _data_path("habitat3d"),
        "parquet_path": _data_path("parquets", "habitat3d_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 319920,
    },
    "gibson": {
        "prefix": _data_path("gibson"),
        "parquet_path": _data_path("parquets", "gibson_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 184560,
    },
    "matterport3d": {
        "prefix": _data_path("matterport3d"),
        "parquet_path": _data_path("parquets", "matterport3d_train.parquet"),
        "root_dir": _data_path("processed_mp3d"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 69600,
    },
    "mp3d": {
        "prefix": _data_path("mp3d"),
        "parquet_path": _data_path("parquets", "mp3d_train.parquet"),
        "tasks": ["depth", "pointmap"],
        "is_video": True,
        "num_images": 193446,
    },
}
