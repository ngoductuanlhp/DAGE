import os

BENCHMARK_DATA_ROOT = "<PATH TO GEOMETRY CRAFTER EVAL DATA>"

BENCHMARK_CONFIGS = {
    "gmu_kitchen": {
        "path": os.path.join(BENCHMARK_DATA_ROOT, "gmu_kitchen"),
        "width": 960,
        "height": 512,
        "use_weight": False,
    },
    "kitti": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "kitti"),
        "width": 768,
        "height": 384,
        "use_weight": True,
    },
    "sintel": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "sintel"),
        "width": 896,
        "height": 448,
        "use_weight": True,
    },
    "monkaa": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "monkaa"),
        "width": 960,
        "height": 512,
        "use_weight": True,
    },
    "ddad": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "ddad"),
        "width": 640,
        "height": 384,
        "use_weight": False,
        "downsample_ratio": 3.0,
    },
    "scannetv2": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "scannetv2"),
        "width": 640,
        "height": 512,
        "use_weight": False,
    },

    "point_odyssey": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "point_odyssey"),
        "width": 640,
        "height": 320,
        "use_weight": False,
    },

    "tartanair": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "tartanair"),
        "width": 640,
        "height": 480,
        "use_weight": False,
    },

    "scannetpp": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "scannetpp"),
        "width": 540,
        "height": 360,
        "use_weight": False,
    },

    "spring": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "spring"),
        "width": 640,
        "height": 360,
        "use_weight": False,
    },


    "waymo": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "waymo"),
        "width": 480,
        "height": 320,
        "use_weight": False,
    },

    "urbansyn": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "urbansyn"),
        "width": 2048,
        "height": 1024,
        "use_weight": False,
    },
    
    "urbansyn_long": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "urbansyn_long"),
        "width": 2048,
        "height": 1024,
        "use_weight": False,
    },
    "unreal4k_quad": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "unreal4k_quad"),
        "width": 960,
        "height": 540,
        "use_weight": False,
    },
    "unreal4k_2k": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "unreal4k_2k"),
        "width": 1920,
        "height": 1080,
        "use_weight": False,
    },
    "unreal4k": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "unreal4k"),
        "width": 3840,
        "height": 2160,
        "use_weight": False,
    },

    "diode": {
        "path":os.path.join(BENCHMARK_DATA_ROOT, "diode"),
        "width": 1024,
        "height": 768,
        "use_weight": False,
    },


}
