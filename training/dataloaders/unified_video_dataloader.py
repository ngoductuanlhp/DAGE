"""
UnifiedVideoDataLoader — simplified single-task mixed-dataset loader.

Supports WeightedConcatDataset for grouping multiple datasets with
weighted sampling, dynamic batch sizes, and MoGe-style augmentation.

Usage:
    from training.dataloaders.unified_video_dataloader import UnifiedVideoDataLoader

    loader = UnifiedVideoDataLoader(config, batch_size=48, num_workers_per_dataset=1, ...)

Config format (YAML):
    datasets:
      dataset: [
        "WeightedConcatDataset(
          VideoDepthCO3DNew(T=10, stride_range=(1, 25)),
          VideoDepthWaymoNew(T=10, stride_range=(1, 8)),
          weights=[94.0, 50.0],
          moge_augmentation=dict(...),
        )"
      ]
      dataset_weights: [1]
      dataset_workers: [10]
"""

import numpy as np
import torch
import random
import re
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from training.dataloaders.batched_sampler import make_sampler
from training.dataloaders.dynamic_batched_sampler import make_dynamic_sampler
from training.dataloaders.weighted_concat_dataset import WeightedConcatDataset


def debug_collate_fn(batch):
    """Custom collate function that reports which key causes errors."""
    if not isinstance(batch[0], dict):
        try:
            return default_collate(batch)
        except RuntimeError as e:
            print(f"Error in collate (non-dict batch): {e}")
            raise

    elem = batch[0]
    collated = {}
    for key in elem:
        try:
            collated[key] = default_collate([d[key] for d in batch])
        except RuntimeError as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Cannot collate key '{key}'")
            print(f"Error message: {e}")
            if isinstance(batch[0][key], torch.Tensor):
                shapes = [d[key].shape for d in batch if isinstance(d[key], torch.Tensor)]
                print(f"  Shapes in batch: {shapes}")
            print(f"{'='*80}\n")
            raise
    return collated


class UnifiedVideoDataLoader:
    def __init__(
        self,
        config,
        batch_size=2,
        num_workers_per_dataset=8,
        shuffle=True,
        pin_memory=True,
        resolutions=None,
        frame_range=None,
        use_moge=False,
        moge_augmentation=None,
        process_index=0,
        use_dynamic_batch_size=False,
    ):
        """Initialize mixed-dataset loader.

        Args:
            config: Dict with keys ``dataset`` (list of dataset strings),
                ``dataset_weights`` (list of floats), and optionally
                ``dataset_workers`` (list of ints).
            batch_size: Default batch size per GPU.
            num_workers_per_dataset: Default worker count per dataloader.
            shuffle: Whether to shuffle.
            pin_memory: Whether to pin memory.
            resolutions: Resolution list for multi-resolution training.
            frame_range: ``[min_frames, max_frames]`` for dynamic batching.
            use_moge: Enable MoGe-style augmentation pipeline.
            moge_augmentation: Global MoGe augmentation config dict.
            process_index: Rank index for distributed training.
            use_dynamic_batch_size: Use dynamic batch sizing based on frame count.
        """
        self.batch_size = batch_size
        self.num_workers_per_dataset = num_workers_per_dataset
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.process_index = process_index
        self.use_moge = use_moge
        self.moge_augmentation = moge_augmentation
        self.use_dynamic_batch_size = use_dynamic_batch_size

        self._set_resolutions(resolutions, use_moge)
        self._set_frame_range(frame_range)

        # Parse config
        dataset_names = config["dataset"]
        dataset_weights = config.get("dataset_weights", [1.0] * len(dataset_names))
        dataset_workers = config.get("dataset_workers", None)

        if len(dataset_names) != len(dataset_weights):
            raise ValueError("Number of datasets and weights must match")
        if dataset_workers is not None and len(dataset_workers) != len(dataset_names):
            raise ValueError("Number of dataset_workers must match number of datasets")

        # Build datasets and dataloaders
        self.datasets = []
        self.dataset_names = []
        self.dataset_weights = dataset_weights
        self.dataloaders = []
        self.epoch = 0

        for ds_idx, ds_name in enumerate(dataset_names):
            num_workers = dataset_workers[ds_idx] if dataset_workers is not None else self.num_workers_per_dataset

            if ds_name.startswith("WeightedConcatDataset"):
                dataset, specific_bs, specific_fr = self._create_weighted_concat_dataset(ds_name)
            else:
                dataset, specific_bs, specific_fr = self._create_regular_dataset(ds_name)

            self.datasets.append(dataset)
            self.dataset_names.append(ds_name)

            dataloader = self._create_dataloader(
                dataset, ds_name,
                specific_batch_size=specific_bs,
                specific_dataset_frame_range=specific_fr,
                num_workers=num_workers,
            )
            self.dataloaders.append(dataloader)

        # Initialise iterators
        self.iterators = {i: iter(dl) for i, dl in enumerate(self.dataloaders)}

    # ------------------------------------------------------------------
    # Dataset creation helpers
    # ------------------------------------------------------------------

    def _create_regular_dataset(self, ds_name):
        import_dir = (
            "training.dataloaders.datasets.image_datasets"
            if ds_name.startswith("Image")
            else "training.dataloaders.datasets.video_datasets_new"
        )
        specific_batch_size = None
        specific_dataset_frame_range = None

        if "(" in ds_name and ")" in ds_name:
            base_name = ds_name.split("(")[0]
            params_str = ds_name.split("(", 1)[1].rsplit(")", 1)[0].strip()
            dataset_module = __import__(import_dir, fromlist=[base_name])
            dataset_class = getattr(dataset_module, base_name)

            if params_str:
                kwargs = eval(f"dict({params_str})")
                if self._resolutions is not None:
                    kwargs["resolutions"] = self._resolutions
                if self.use_moge:
                    kwargs["use_moge"] = True
                    if "moge_augmentation" not in kwargs:
                        assert self.moge_augmentation is not None
                        kwargs["moge_augmentation"] = self.moge_augmentation
                    else:
                        specific_batch_size, specific_dataset_frame_range = (
                            self._batch_size_from_area(kwargs["moge_augmentation"])
                        )
                if self.process_index == 0:
                    print(f"Creating dataset: {base_name}")
                dataset = dataset_class(**kwargs)
            else:
                kwargs = {}
                if self._resolutions is not None:
                    kwargs["resolutions"] = self._resolutions
                if self.use_moge:
                    kwargs["use_moge"] = True
                    if self.moge_augmentation is not None:
                        kwargs["moge_augmentation"] = self.moge_augmentation
                dataset = dataset_class(**kwargs)
        else:
            dataset_module = __import__(import_dir, fromlist=[ds_name])
            dataset_class = getattr(dataset_module, ds_name)
            dataset = dataset_class()

        return dataset, specific_batch_size, specific_dataset_frame_range

    def _create_weighted_concat_dataset(self, ds_name):
        params_str = ds_name.split("(", 1)[1].rsplit(")", 1)[0].strip()
        specific_batch_size = None
        specific_dataset_frame_range = None

        # Tokenize top-level comma-separated parts
        parts = []
        current = []
        depth = 0
        for char in params_str:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(char)
        if current:
            parts.append("".join(current).strip())

        dataset_defs = []
        shared_params = {}
        for part in parts:
            if "=" in part and not part.strip().startswith("Video") and not part.strip().startswith("Image"):
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if value.startswith("dict("):
                    value = re.sub(r"(\w+)(\s*):", r"\1\2=", value)
                    value = value.replace("false", "False").replace("true", "True")
                shared_params[key] = eval(value)
            else:
                dataset_defs.append(part)

        # Determine batch size / frame range from shared moge_augmentation
        if "moge_augmentation" in shared_params and self.use_moge:
            if shared_params.get("specific_batch_size") is not None and shared_params.get("specific_dataset_frame_range") is not None:
                specific_batch_size = shared_params["specific_batch_size"]
                specific_dataset_frame_range = shared_params["specific_dataset_frame_range"]
            else:
                specific_batch_size, specific_dataset_frame_range = (
                    self._batch_size_from_area(shared_params["moge_augmentation"])
                )
            if self.process_index == 0:
                print(
                    f"WeightedConcatDataset group: batch_size={specific_batch_size}, "
                    f"frame_range={specific_dataset_frame_range}"
                )

        # Build child datasets
        datasets = []
        for ds_def in dataset_defs:
            import_dir = (
                "training.dataloaders.datasets.image_datasets"
                if ds_def.startswith("Image")
                else "training.dataloaders.datasets.video_datasets_new"
            )
            base_name = ds_def.split("(")[0]
            ds_params_str = ds_def.split("(", 1)[1].rsplit(")", 1)[0].strip() if "(" in ds_def else ""
            dataset_module = __import__(import_dir, fromlist=[base_name])
            dataset_class = getattr(dataset_module, base_name)

            ds_kwargs = eval(f"dict({ds_params_str})") if ds_params_str else {}
            if self._resolutions is not None:
                ds_kwargs["resolutions"] = self._resolutions
            if self.use_moge:
                ds_kwargs["use_moge"] = True
                if "moge_augmentation" not in ds_kwargs:
                    if "moge_augmentation" in shared_params:
                        ds_kwargs["moge_augmentation"] = shared_params["moge_augmentation"]
                    elif self.moge_augmentation is not None:
                        ds_kwargs["moge_augmentation"] = self.moge_augmentation

            if self.process_index == 0:
                print(f"  Creating child dataset: {base_name}")
            datasets.append(dataset_class(**ds_kwargs))

        weights = shared_params.get("weights", [1.0] * len(datasets))
        weighted_dataset = WeightedConcatDataset(*datasets, weights=weights)
        if self.process_index == 0:
            print(f"Created WeightedConcatDataset with {len(datasets)} datasets, weights: {weights}")
        return weighted_dataset, specific_batch_size, specific_dataset_frame_range

    # ------------------------------------------------------------------
    # DataLoader creation
    # ------------------------------------------------------------------

    def _create_dataloader(self, dataset, ds_name, specific_batch_size=None, specific_dataset_frame_range=None, num_workers=None):
        if num_workers is None:
            num_workers = self.num_workers_per_dataset

        base_ds_name = ds_name.split("(")[0] if "(" in ds_name else ds_name
        if self.process_index == 0:
            print(f"  Using {num_workers} workers for: {base_ds_name}")

        number_of_resolutions = len(dataset.resolutions) if dataset.resolutions is not None else None
        dataset_frame_range = self._frame_range

        if self.use_dynamic_batch_size:
            if any(name in ds_name for name in ["VideoDepthHabitat3DNew", "VideoDepthGibsonNew", "VideoDepthMatterport3DNew"]):
                sampler = make_sampler(
                    dataset, batch_size=1,
                    number_of_resolutions=number_of_resolutions,
                    min_num_frames=1, max_num_frames=1,
                    shuffle=self.shuffle, drop_last=True,
                    process_index=self.process_index,
                )
            else:
                sampler = make_dynamic_sampler(
                    dataset,
                    target_total_views=specific_batch_size or self.batch_size,
                    number_of_resolutions=number_of_resolutions,
                    min_num_frames=(specific_dataset_frame_range or dataset_frame_range)[0],
                    max_num_frames=(specific_dataset_frame_range or dataset_frame_range)[1],
                    shuffle=self.shuffle,
                    process_index=self.process_index,
                )
        else:
            sampler = make_sampler(
                dataset, batch_size=self.batch_size,
                number_of_resolutions=number_of_resolutions,
                min_num_frames=dataset_frame_range[0],
                max_num_frames=dataset_frame_range[1],
                shuffle=self.shuffle, drop_last=True,
                process_index=self.process_index,
            )

        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            collate_fn=debug_collate_fn,
        )

        if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "set_epoch"):
            dataloader.dataset.set_epoch(0)
        if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(0)
        if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "sampler") and hasattr(dataloader.batch_sampler.sampler, "set_epoch"):
            dataloader.batch_sampler.sampler.set_epoch(0)

        return dataloader

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            # Sample a dataset based on weights
            dataset_idx = random.choices(
                range(len(self.datasets)),
                weights=self.dataset_weights,
                k=1,
            )[0]

            try:
                try:
                    batch = next(self.iterators[dataset_idx])
                except StopIteration:
                    self.epoch += 1
                    dl = self.dataloaders[dataset_idx]
                    if hasattr(dl, "dataset") and hasattr(dl.dataset, "set_epoch"):
                        dl.dataset.set_epoch(self.epoch)
                    if hasattr(dl, "batch_sampler") and hasattr(dl.batch_sampler, "set_epoch"):
                        dl.batch_sampler.set_epoch(self.epoch)
                    if hasattr(dl, "batch_sampler") and hasattr(dl.batch_sampler, "sampler") and hasattr(dl.batch_sampler.sampler, "set_epoch"):
                        dl.batch_sampler.sampler.set_epoch(self.epoch)
                    self.iterators[dataset_idx] = iter(dl)
                    batch = next(self.iterators[dataset_idx])

                if batch is None or not batch:
                    continue

                # Skip invalid batches
                if "valid" in batch:
                    if isinstance(batch["valid"], torch.Tensor):
                        if not torch.all(batch["valid"]):
                            continue
                    elif isinstance(batch["valid"], list):
                        if not all(batch["valid"]):
                            continue
                    elif not batch["valid"]:
                        continue

                # Add dataset info
                if "dataset" not in batch:
                    batch["dataset"] = self.dataset_names[dataset_idx]

                return batch
            except StopIteration:
                raise StopIteration

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_size_from_area(moge_aug):
        """Determine batch size and frame range from MoGe area_range."""
        max_area = moge_aug["area_range"][1]
        if max_area <= 255000:
            return 44, [2, 24]
        elif max_area <= 600000:
            return 24, [4, 6]
        else:
            return 16, [4, 4]

    def _set_resolutions(self, resolutions, use_moge=False):
        if use_moge:
            self._resolutions = [[0, 1]] * 10000
            return
        if resolutions is None:
            self._resolutions = None
            return
        self._resolutions = []
        for r in resolutions:
            if isinstance(r, int):
                self._resolutions.append((r, r))
            elif isinstance(r, (tuple, list)):
                self._resolutions.append((r[0], r[1]))

    def _set_frame_range(self, frame_range):
        self._frame_range = frame_range if frame_range is not None else [8, 16]
