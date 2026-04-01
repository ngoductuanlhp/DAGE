"""
UnifiedVideoDataLoaderV3 with WeightedConcatDataset support

This version adds support for grouping multiple datasets using WeightedConcatDataset
to reduce memory overhead. See WEIGHTED_CONCAT_USAGE.md for usage examples.

Key differences from V1:
- Supports WeightedConcatDataset(...) syntax in config
- Significantly reduces memory when grouping datasets
- Backward compatible with original syntax

Usage:
    from training.dataloaders.unified_video_dataloader_v3 import UnifiedVideoDataLoaderV3
    
    loader = UnifiedVideoDataLoaderV3(config, batch_size=48, num_workers_per_dataset=1, ...)
"""

import numpy as np
import torch
import random
import re
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from training.dataloaders.config import DATASET_CONFIG

from training.dataloaders.batched_sampler import make_sampler
from training.dataloaders.dynamic_batched_sampler import make_dynamic_sampler
from training.dataloaders.weighted_concat_dataset import WeightedConcatDataset


def debug_collate_fn(batch):
    """
    Custom collate function for debugging that reports which key causes resize errors.
    Identical to default_collate but with error reporting.
    """
    if not isinstance(batch[0], dict):
        # Not a dict, use default collate
        try:
            return default_collate(batch)
        except RuntimeError as e:
            print(f"Error in collate (non-dict batch): {e}")
            print(f"Batch type: {type(batch[0])}")
            raise
    
    # For dict batches, collate each key separately to identify the problematic one
    elem = batch[0]
    collated = {}
    
    for key in elem:
        try:
            collated[key] = default_collate([d[key] for d in batch])
        except RuntimeError as e:
            print(f"\n{'='*80}")
            print(f"ERROR: Cannot collate key '{key}'")
            print(f"Error message: {e}")
            print(f"Key '{key}' details:")
            print(f"  - Type: {type(batch[0][key])}")
            if isinstance(batch[0][key], torch.Tensor):
                print(f"  - Shape: {batch[0][key].shape}")
                print(f"  - Dtype: {batch[0][key].dtype}")
                print(f"  - Device: {batch[0][key].device}")
                print(f"  - Is contiguous: {batch[0][key].is_contiguous()}")
                print(f"  - Requires grad: {batch[0][key].requires_grad}")
                print(f"  - Storage ptr: {batch[0][key].storage().data_ptr()}")
                
                # Check all samples in batch
                shapes = [d[key].shape if isinstance(d[key], torch.Tensor) else 'not tensor' for d in batch]
                print(f"  - Shapes in batch: {shapes}")
                
                # Try to identify if it's a memory issue
                try:
                    test_tensor = batch[0][key].clone()
                    print(f"  - Can clone: Yes")
                except:
                    print(f"  - Can clone: No")
            print(f"{'='*80}\n")
            raise
    
    return collated


class UnifiedVideoDataLoaderV3:
    def __init__(self, config, batch_size=2, num_workers_per_dataset=8, shuffle=True, pin_memory=True, resolutions=None, frame_range=None, use_moge=False, moge_augmentation=None, process_index=0, use_dynamic_batch_size=False):
        """Initialize MixedDataLoader with a task/dataset config structure.
        
        This version supports WeightedConcatDataset for grouping datasets.
        
        Args:
            config (dict): Dict of tasks with their weights and datasets
            batch_size (int): Batch size for the data loader
            num_workers_per_dataset (int): Number of workers for each data loader
            shuffle (bool): Whether to shuffle the data loader
            pin_memory (bool): Whether to pin memory for the data loader
            process_index (int): Index of the current process in distributed training
        
        Example config with WeightedConcatDataset:
            {
                'video_pointmap': {
                    'task_weight': 1.0,
                    'dataset': [
                        "WeightedConcatDataset(
                            VideoDepthCO3DNew(T=10, stride_range=(1,25)),
                            VideoDepthWildRGBDNew(T=10, stride_range=(1,25)),
                            weights=[1.0, 1.0],
                            moge_augmentation=dict(...)
                        )",
                        "VideoDepthStandaloneNew(...)",  # Can mix with regular datasets
                    ],
                    'dataset_weights': [1.0, 1.0]
                }
            }
        """

        self.config = config
        self.batch_size = batch_size
        self.num_workers_per_dataset = num_workers_per_dataset
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.process_index = process_index

        self.use_moge = use_moge
        self.moge_augmentation = moge_augmentation
        self.use_dynamic_batch_size = use_dynamic_batch_size
        
        # Get list of all tasks
        self.tasks = list(config.keys())
        
        # Get task weights
        self.task_weights = [config[task]['task_weight'] for task in self.tasks]
        
        # Initialize all dataloaders for each task and dataset
        self.task_datasets = {}
        self.dataloaders = {}

        self._set_resolutions(resolutions=resolutions, use_moge=self.use_moge)
        self._set_frame_range(frame_range)
        
        for task in self.tasks:
            task_config = config[task]
            
            # Get dataset names and weights for this task
            dataset_names = task_config['dataset']
            dataset_weights = task_config.get('dataset_weights', [1.0] * len(dataset_names))
            dataset_workers = task_config.get('dataset_workers', None)  # Optional per-dataset worker counts
            
            # Ensure dataset names and weights match in length
            if len(dataset_names) != len(dataset_weights):
                raise ValueError(f"Number of datasets and weights must match for task {task}")
            
            # If dataset_workers is provided, ensure it matches dataset count
            if dataset_workers is not None and len(dataset_workers) != len(dataset_names):
                raise ValueError(f"Number of dataset_workers must match number of datasets for task {task}")
            
            # Initialize datasets for this task
            self.task_datasets[task] = {
                'datasets': [],
                'dataset_names': [],
                'weights': dataset_weights,
                'workers': dataset_workers,  # Store worker counts for reference
                'dataloaders': [],
                'epoch': 0
            }

            for ds_idx, ds_name in enumerate(dataset_names):
                # Import and create dataset instance based on dataset name
                try:
                    # Get worker count for this specific dataset (if provided)
                    num_workers = dataset_workers[ds_idx] if dataset_workers is not None else self.num_workers_per_dataset
                    
                    # Special handling for WeightedConcatDataset
                    if ds_name.startswith('WeightedConcatDataset'):
                        dataset, specific_batch_size, specific_dataset_frame_range = self._create_weighted_concat_dataset(ds_name, task)
                        
                        # Store dataset and its dataloader
                        self.task_datasets[task]['datasets'].append(dataset)
                        self.task_datasets[task]['dataset_names'].append(ds_name)
                        
                        # Create dataloader for this dataset, using group-specific batch_size/frame_range and workers
                        dataloader = self._create_dataloader_for_dataset(
                            dataset, ds_name,
                            specific_batch_size=specific_batch_size,
                            specific_dataset_frame_range=specific_dataset_frame_range,
                            num_workers=num_workers
                        )
                        self.task_datasets[task]['dataloaders'].append(dataloader)
                        continue
                    
                    # Regular dataset handling (same as V1)
                    dataset, specific_batch_size, specific_dataset_frame_range = self._create_regular_dataset(ds_name, task)
                    
                    # Create DataLoader for this dataset
                    base_ds_name = ds_name.split('(')[0] if '(' in ds_name else ds_name
                    dataloader = self._create_dataloader_for_dataset(
                        dataset, ds_name, 
                        specific_batch_size=specific_batch_size,
                        specific_dataset_frame_range=specific_dataset_frame_range,
                        num_workers=num_workers
                    )
                    
                    # Store dataset and its dataloader
                    self.task_datasets[task]['datasets'].append(dataset)
                    self.task_datasets[task]['dataset_names'].append(ds_name)
                    self.task_datasets[task]['dataloaders'].append(dataloader)
                    
                except (ImportError, AttributeError) as e:
                    raise ValueError(f"Could not find dataset: {ds_name}, error: {e}")
        
        # Dictionary to store iterators for each dataloader
        self.iterators = {}
        
        # Initialize all iterators
        for task in self.tasks:
            selected_task_datasets = self.task_datasets[task]
            for dataset_idx in range(len(selected_task_datasets['datasets'])):
                key = (task, dataset_idx)
                dataset_name = selected_task_datasets['dataset_names'][dataset_idx]
                
                self.iterators[key] = iter(selected_task_datasets['dataloaders'][dataset_idx])

    def _create_regular_dataset(self, ds_name, task):
        """Create a regular (non-WeightedConcat) dataset"""
        import_dir = 'training.dataloaders.datasets.image_datasets' if ds_name.startswith('Image') else 'training.dataloaders.datasets.video_datasets_new'
        
        specific_batch_size = None
        specific_dataset_frame_range = None
        
        # Parse dataset name and parameters
        if '(' in ds_name and ')' in ds_name:
            # Parse the base name and parameters
            base_name = ds_name.split('(')[0]
            params_str = ds_name.split('(', 1)[1].rsplit(')', 1)[0].strip()
            
            # Import dataset class
            dataset_module = __import__(import_dir, fromlist=[base_name])
            dataset_class = getattr(dataset_module, base_name)
            
            # Check if parameters are empty
            if params_str:
                # Evaluate the parameters string to get kwargs
                kwargs = eval(f"dict({params_str})")

                if self._resolutions is not None:
                    kwargs['resolutions'] = self._resolutions

                if self.use_moge:
                    kwargs['use_moge'] = self.use_moge
                    # Only inject global moge_augmentation if provided and not already set per-dataset
                    if 'moge_augmentation' not in kwargs:
                        assert self.moge_augmentation is not None, "moge_augmentation is not provided"
                        kwargs['moge_augmentation'] = self.moge_augmentation
                    else:
                        # NOTE recalculate here
                        specific_max_area = kwargs['moge_augmentation']['area_range'][1]
                        specific_min_area = kwargs['moge_augmentation']['area_range'][0]    

                        if specific_max_area <= 255000:
                            specific_batch_size = 44
                            specific_dataset_frame_range = [2, 24]
                        elif specific_max_area <= 600000:
                            specific_batch_size = 24
                            specific_dataset_frame_range = [4, 6]
                        else:
                            specific_batch_size = 16
                            specific_dataset_frame_range = [4, 4]
                        
                        if self.process_index == 0:
                            print(f"dataset name: {base_name}, specific_batch_size: {specific_batch_size}, specific_dataset_frame_range: {specific_dataset_frame_range}, area range: {specific_max_area} - {specific_min_area}")

                if self.process_index == 0:
                    if self.use_moge:
                        kwargs_print = kwargs.copy()
                        kwargs_print.pop('resolutions')
                        print(f"Creating dataset: {base_name}, {kwargs_print}")
                    else:
                        print(f"Creating dataset: {base_name}, {kwargs}")

                # Create dataset with parameters
                dataset = dataset_class(**kwargs)
            else:
                # Empty parentheses, create dataset without parameters
                if self.use_moge:
                    if self._resolutions is not None:
                        if self.moge_augmentation is not None:
                            dataset = dataset_class(resolutions=self._resolutions, use_moge=self.use_moge, moge_augmentation=self.moge_augmentation)
                        else:
                            dataset = dataset_class(resolutions=self._resolutions, use_moge=self.use_moge)
                    else:
                        if self.moge_augmentation is not None:
                            dataset = dataset_class(use_moge=self.use_moge, moge_augmentation=self.moge_augmentation)
                        else:
                            dataset = dataset_class(use_moge=self.use_moge)
                elif self._resolutions is not None:
                    dataset = dataset_class(resolutions=self._resolutions)
                else:
                    dataset = dataset_class()
        else:
            # No parentheses, import and create dataset without parameters
            dataset_module = __import__(import_dir, fromlist=[ds_name])
            dataset_class = getattr(dataset_module, ds_name)
            dataset = dataset_class()
        
        return dataset, specific_batch_size, specific_dataset_frame_range

    def _create_weighted_concat_dataset(self, ds_name, task):
        """
        Create a WeightedConcatDataset from a config string.
        
        Shared parameters (like weights, moge_augmentation) are applied to ALL datasets.
        
        Returns:
            tuple: (dataset, specific_batch_size, specific_dataset_frame_range)
        
        Example:
            "WeightedConcatDataset(
                VideoDepthCO3DNew(T=10, stride_range=(1,25)),
                VideoDepthWildRGBDNew(T=10, stride_range=(1,25)),
                weights=[1.0, 1.0],
                moge_augmentation=dict(area_range=[49152, 150000], ...)
            )"
            
        In the above example:
        - weights=[1.0, 1.0] is used by WeightedConcatDataset for sampling
        - moge_augmentation=dict(...) is applied to BOTH VideoDepthCO3DNew and VideoDepthWildRGBDNew
        - The area_range is used to calculate batch_size and frame_range for the entire group
        """
        # Extract parameters from the config string
        params_str = ds_name.split('(', 1)[1].rsplit(')', 1)[0].strip()
        
        # These will be calculated from shared moge_augmentation
        specific_batch_size = None
        specific_dataset_frame_range = None
        
        # Parse datasets and parameters
        parts = []
        current = []
        paren_depth = 0
        bracket_depth = 0
        
        for char in params_str:
            if char == '(' or char == '[' or char == '{':
                paren_depth += 1
                bracket_depth += (1 if char == '[' else 0)
            elif char == ')' or char == ']' or char == '}':
                paren_depth -= 1
                bracket_depth -= (1 if char == ']' else 0)
            elif char == ',' and paren_depth == 0 and bracket_depth == 0:
                parts.append(''.join(current).strip())
                current = []
                continue
            current.append(char)
        
        if current:
            parts.append(''.join(current).strip())
        
        # Separate dataset definitions from parameters
        dataset_defs = []
        shared_params = {}
        
        for part in parts:
            if '=' in part and not part.strip().startswith('Video') and not part.strip().startswith('Image'):
                # This is a parameter
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle YAML-style dict syntax (key: value) -> Python syntax (key=value)
                # This is needed because YAML configs may use dict(key: value, ...) 
                # but Python eval expects dict(key=value, ...)
                if value.startswith('dict('):
                    # Convert YAML-style colons to Python-style equals inside dict()
                    # Match pattern: word_characters followed by colon (but not inside quotes or brackets)
                    value = re.sub(r'(\w+)(\s*):', r'\1\2=', value)
                    # Convert YAML booleans to Python booleans
                    value = value.replace('false', 'False').replace('true', 'True')
                
                shared_params[key] = eval(value)
            else:
                # This is a dataset definition
                dataset_defs.append(part)
        
        # Calculate specific_batch_size and specific_dataset_frame_range from shared moge_augmentation
        if 'moge_augmentation' in shared_params and self.use_moge:
            specific_max_area = shared_params['moge_augmentation']['area_range'][1]
            specific_min_area = shared_params['moge_augmentation']['area_range'][0]

            if shared_params.get('specific_batch_size', None) is None or shared_params.get('specific_dataset_frame_range', None) is None:
            
                if specific_max_area <= 255000:
                    specific_batch_size = 44
                    specific_dataset_frame_range = [2, 24]
                elif specific_max_area <= 600000:
                    specific_batch_size = 24
                    specific_dataset_frame_range = [4, 6]
                else:
                    specific_batch_size = 16
                    specific_dataset_frame_range = [4, 4]
            else:
                specific_batch_size = shared_params['specific_batch_size']
                specific_dataset_frame_range = shared_params['specific_dataset_frame_range']
            
            if self.process_index == 0:
                print(f"WeightedConcatDataset group config: specific_batch_size={specific_batch_size}, specific_dataset_frame_range={specific_dataset_frame_range}, area_range={specific_min_area}-{specific_max_area}")
        
        # Create individual datasets
        datasets = []
        for ds_def in dataset_defs:
            # Check if it's Image or Video dataset
            import_dir = 'training.dataloaders.datasets.image_datasets' if ds_def.startswith('Image') else 'training.dataloaders.datasets.video_datasets_new'
            
            # Parse dataset name
            base_name = ds_def.split('(')[0]
            ds_params_str = ds_def.split('(', 1)[1].rsplit(')', 1)[0].strip() if '(' in ds_def else ''
            
            # Import dataset class
            dataset_module = __import__(import_dir, fromlist=[base_name])
            dataset_class = getattr(dataset_module, base_name)
            
            # Parse dataset-specific parameters
            if ds_params_str:
                ds_kwargs = eval(f"dict({ds_params_str})")
            else:
                ds_kwargs = {}
            
            # Add shared parameters (resolutions, moge_augmentation, etc.)
            if self._resolutions is not None:
                ds_kwargs['resolutions'] = self._resolutions
            
            if self.use_moge:
                ds_kwargs['use_moge'] = self.use_moge
                # Apply moge_augmentation in priority order:
                # 1. Per-dataset (if dataset has its own in parentheses) - already in ds_kwargs
                # 2. Shared from WeightedConcatDataset config (most common case)
                # 3. Global from dataloader constructor
                if 'moge_augmentation' not in ds_kwargs:
                    # No per-dataset moge_augmentation specified
                    if 'moge_augmentation' in shared_params:
                        # Use shared moge_augmentation from WeightedConcatDataset config
                        ds_kwargs['moge_augmentation'] = shared_params['moge_augmentation']
                    elif self.moge_augmentation is not None:
                        # Fall back to global moge_augmentation
                        ds_kwargs['moge_augmentation'] = self.moge_augmentation
                
                # Log dataset creation (batch_size/frame_range are now handled at group level)
                if self.process_index == 0:
                    print(f"  Creating dataset in WeightedConcatDataset: {base_name}")
            else:
                if self.process_index == 0:
                    print(f"  Creating dataset in WeightedConcatDataset: {base_name}")
            
            # Create dataset
            dataset = dataset_class(**ds_kwargs)
            datasets.append(dataset)
        
        # Extract weights from shared_params
        weights = shared_params.get('weights', [1.0] * len(datasets))
        
        # Create WeightedConcatDataset
        weighted_dataset = WeightedConcatDataset(*datasets, weights=weights)
        
        if self.process_index == 0:
            print(f"Created WeightedConcatDataset with {len(datasets)} datasets, weights: {weights}")
        
        return weighted_dataset, specific_batch_size, specific_dataset_frame_range
    
    def _create_dataloader_for_dataset(self, dataset, ds_name, specific_batch_size=None, specific_dataset_frame_range=None, num_workers=None):
        """Create a DataLoader for a given dataset
        
        Args:
            dataset: The dataset instance
            ds_name: Name/config string of the dataset
            specific_batch_size: Override batch size for this dataset (optional)
            specific_dataset_frame_range: Override frame range for this dataset (optional)
            num_workers: Number of workers for this dataset (optional, uses self.num_workers_per_dataset if None)
        """
        # Use provided num_workers or fall back to global setting
        if num_workers is None:
            num_workers = self.num_workers_per_dataset
        
        base_ds_name = ds_name.split('(')[0] if '(' in ds_name else ds_name
        
        # Log worker configuration for visibility
        if self.process_index == 0:
            if num_workers != self.num_workers_per_dataset:
                print(f"  Using {num_workers} workers for dataset/group: {base_ds_name} (overriding default {self.num_workers_per_dataset})")
            else:
                print(f"  Using {num_workers} workers (default) for dataset/group: {base_ds_name}")
        
        number_of_resolutions = len(dataset.resolutions) if dataset.resolutions is not None else None

        
        dataset_frame_range = self._frame_range
        
        if self.use_dynamic_batch_size:
            if any(special_dataset in ds_name for special_dataset in ["VideoDepthHabitat3DNew", "VideoDepthGibsonNew", "VideoDepthMatterport3DNew"]):
                dataloader_sampler = make_sampler(
                    dataset,
                    batch_size=1,
                    number_of_resolutions=number_of_resolutions,
                    min_num_frames=1,
                    max_num_frames=1,
                    shuffle=self.shuffle,
                    drop_last=True,
                    process_index=self.process_index
                )
            else:
                dataloader_sampler = make_dynamic_sampler(
                    dataset,
                    target_total_views=specific_batch_size if specific_batch_size is not None else self.batch_size,
                    number_of_resolutions=number_of_resolutions,
                    min_num_frames=specific_dataset_frame_range[0] if specific_dataset_frame_range is not None else dataset_frame_range[0],
                    max_num_frames=specific_dataset_frame_range[1] if specific_dataset_frame_range is not None else dataset_frame_range[1],
                    shuffle=self.shuffle,
                    process_index=self.process_index
                )
        else:
            dataloader_sampler = make_sampler(
                dataset,
                batch_size=self.batch_size,
                number_of_resolutions=number_of_resolutions,
                min_num_frames=dataset_frame_range[0],
                max_num_frames=dataset_frame_range[1],
                shuffle=self.shuffle,
                drop_last=True,
                process_index=self.process_index
            )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=dataloader_sampler,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            collate_fn=debug_collate_fn,
        )
        
        if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "set_epoch"):
            dataloader.dataset.set_epoch(0)
        if hasattr(dataloader, "batch_sampler") and hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(0)
        if (
            hasattr(dataloader, "batch_sampler")
            and hasattr(dataloader.batch_sampler, "sampler")
            and hasattr(dataloader.batch_sampler.sampler, "set_epoch")
        ):
            dataloader.batch_sampler.sampler.set_epoch(0)
        
        return dataloader

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            # Sample a task based on task weights
            current_task = random.choices(
                self.tasks, 
                weights=self.task_weights, 
                k=1
            )[0]
            
            # Get task info
            selected_task_datasets = self.task_datasets[current_task]
            
            # Sample a dataset based on dataset weights
            dataset_idx = random.choices(
                range(len(selected_task_datasets['datasets'])), 
                weights=selected_task_datasets['weights'], 
                k=1
            )[0]
            
            # Create a key for this task-dataset combination
            key = (current_task, dataset_idx)
            
            # Try to get a batch from the current loader
            try:
                try:
                    # Try to get the next batch from the existing iterator
                    batch = next(self.iterators[key])
                except StopIteration:
                    new_epoch = selected_task_datasets['epoch'] + 1
                    selected_task_datasets['epoch'] = new_epoch
                    if hasattr(selected_task_datasets['dataloaders'][dataset_idx], "dataset") and hasattr(selected_task_datasets['dataloaders'][dataset_idx].dataset, "set_epoch"):
                        selected_task_datasets['dataloaders'][dataset_idx].dataset.set_epoch(new_epoch)
                    if hasattr(selected_task_datasets['dataloaders'][dataset_idx], "batch_sampler") and hasattr(selected_task_datasets['dataloaders'][dataset_idx].batch_sampler, "set_epoch"):
                        selected_task_datasets['dataloaders'][dataset_idx].batch_sampler.set_epoch(new_epoch)
                    if (
                        hasattr(selected_task_datasets['dataloaders'][dataset_idx], "batch_sampler")
                        and hasattr(selected_task_datasets['dataloaders'][dataset_idx].batch_sampler, "sampler")
                        and hasattr(selected_task_datasets['dataloaders'][dataset_idx].batch_sampler.sampler, "set_epoch")
                    ):
                        selected_task_datasets['dataloaders'][dataset_idx].batch_sampler.sampler.set_epoch(new_epoch)
                    # If iterator is exhausted, create a new one
                    self.iterators[key] = iter(selected_task_datasets['dataloaders'][dataset_idx])
                    batch = next(self.iterators[key])
                
                # Skip if batch is None or empty
                if batch is None or not batch:
                    continue
                
                # Check valid flag in batch
                if 'valid' in batch:
                    if isinstance(batch['valid'], (list, torch.Tensor)):
                        # Multi-sample batch: skip if any sample is invalid
                        if isinstance(batch['valid'], torch.Tensor):
                            if not torch.all(batch['valid']):
                                continue
                        else:
                            if not all(batch['valid']):
                                continue
                    else:
                        # Single-sample batch
                        if not batch['valid']:
                            continue

                # Add task information to the batch
                batch['task'] = current_task
                
                # Add dataset information to the batch
                current_dataset_name = selected_task_datasets['dataset_names'][dataset_idx]
                
                # Check if 'dataset' is already set by WeightedConcatDataset (individual dataset name)
                if 'dataset' in batch:
                    # Preserve the individual dataset name, add group name separately
                    batch['dataset_group'] = current_dataset_name
                else:
                    # Regular dataset, just set the dataset name
                    batch['dataset'] = current_dataset_name
                
                return batch
            except StopIteration:
                # This should only happen if a dataloader is completely empty
                raise StopIteration

    def _set_resolutions(self, resolutions, use_moge=False):
        # NOTE for MoGe, we use a dummy list of 10000 resolutions
        if use_moge:
            self._resolutions = [[0,1]] * 10000
            return
        
        if resolutions is None:
            self._resolutions = None
            return
        
        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            elif isinstance(resolution, tuple):
                width, height = resolution
            else:
                width, height = resolution[0], resolution[1]
            
            assert isinstance(width, int), f"Bad type for {width=} {type(width)=}, should be int"
            assert isinstance(height, int), f"Bad type for {height=} {type(height)=}, should be int"
            self._resolutions.append((width, height))
    
    def _set_frame_range(self, frame_range):
        if frame_range is None:
            self._frame_range = [8, 16]
        else:
            self._frame_range = frame_range

