import numpy as np
import random
from torch.utils.data import Dataset


class WeightedConcatDataset(Dataset):
    """
    A dataset that concatenates multiple datasets and samples from them based on weights.
    
    This is more memory-efficient than creating separate DataLoaders for each dataset,
    as it allows using a single DataLoader with shared worker processes.
    
    Can be used in two ways:
    1. Direct instantiation: WeightedConcatDataset(datasets=[...], weights=[...], dataset_names=[...])
    2. Config-style (for UnifiedVideoDataLoader): Pass dataset classes as *args with shared kwargs
    
    Args:
        *args: Variable number of dataset class names (strings) or instances
        datasets (list, optional): List of dataset instances (if not using *args)
        weights (list, optional): List of sampling weights for each dataset (will be normalized)
        dataset_names (list, optional): List of dataset names for tracking
        lengths (list, optional): Pre-computed lengths for each dataset. If None, will compute.
        **kwargs: Shared keyword arguments to pass to all datasets (resolutions, moge_augmentation, etc.)
    
    Example usage in config:
        "WeightedConcatDataset(
            VideoDepthCO3DNew(T=10, stride_range=(1,25)),
            VideoDepthWildRGBDNew(T=10, stride_range=(1,25)),
            weights=[1.0, 1.0],
            moge_augmentation=dict(...)
        )"
    """
    
    def __init__(self, *args, datasets=None, weights=None, dataset_names=None, lengths=None, **kwargs):
        # Mode 1: Using *args (config-style initialization)
        if args and datasets is None:
            # args contains dataset instances that were already created
            self.datasets = list(args)
            self.num_datasets = len(self.datasets)
            
            # Extract dataset names from instances
            if dataset_names is None:
                self.dataset_names = [type(ds).__name__ for ds in self.datasets]
            else:
                self.dataset_names = dataset_names
            
            # Use provided weights or default to equal weights
            if weights is None:
                weights = [1.0] * self.num_datasets
        
        # Mode 2: Traditional initialization with datasets list
        elif datasets is not None:
            assert len(datasets) == len(weights) == len(dataset_names), \
                "datasets, weights, and dataset_names must have the same length"
            
            self.datasets = datasets
            self.dataset_names = dataset_names
            self.num_datasets = len(datasets)
        
        else:
            raise ValueError("Either provide datasets as *args or as datasets= parameter")
        
        # Normalize weights to probabilities
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
        # Compute or use provided lengths
        if lengths is None:
            self.lengths = [len(ds) for ds in self.datasets]
        else:
            self.lengths = lengths
        
        self.total_length = sum(self.lengths)
        
        # For tracking which dataset was sampled (useful for debugging)
        self._last_sampled_dataset_idx = None
        
        # Epoch tracking for distributed training
        self.epoch = 0
        
        # Expose resolutions attribute from first dataset (for compatibility with UnifiedVideoDataLoader)
        # This assumes all concatenated datasets share the same resolutions
        if self.datasets and hasattr(self.datasets[0], 'resolutions'):
            self.resolutions = self.datasets[0].resolutions
        else:
            self.resolutions = None
    
    def __len__(self):
        """
        Return total length. Note: This is approximate since we sample by weight,
        not by concatenation. For infinite sampling, you might want to return a large number.
        """
        return self.total_length
    
    def __getitem__(self, idx):
        """
        Sample a dataset based on weights, then sample from that dataset.
        
        Args:
            idx: Index from the sampler. Can be:
                - int: simple index
                - tuple: (sample_idx, view_size, feat_idx) from dynamic sampler - MUST preserve feat_idx for resolution consistency!
        
        Returns:
            dict: Sample from the selected dataset with added metadata
        """
        # Sample which dataset to use based on weights
        dataset_idx = random.choices(
            range(self.num_datasets),
            weights=self.weights,
            k=1
        )[0]
        
        self._last_sampled_dataset_idx = dataset_idx
        
        # Handle different index formats from the sampler
        if isinstance(idx, tuple):
            # Dynamic sampler format: (sample_idx, view_size, feat_idx) or (sample_idx, view_size)
            # IMPORTANT: We must pass the tuple with the same feat_idx to maintain resolution consistency!
            # But we use random sampling within the selected dataset to match original behavior
            sample_idx_random = random.randint(0, self.lengths[dataset_idx] - 1)
            
            # Reconstruct the index tuple with random sample index but SAME feat_idx
            if len(idx) == 3:
                mapped_idx = (sample_idx_random, idx[1], idx[2])  # Preserve feat_idx!
            else:
                mapped_idx = (sample_idx_random, idx[1])
            
            try:
                sample = self.datasets[dataset_idx][mapped_idx]
            except Exception as e:
                print(f"Warning: Failed to get sample {mapped_idx} from dataset {self.dataset_names[dataset_idx]}: {e}")
                # Fallback: try with just the random sample index
                try:
                    sample = self.datasets[dataset_idx][sample_idx_random]
                except:
                    # Last resort: another random sample
                    sample_idx_fallback = random.randint(0, self.lengths[dataset_idx] - 1)
                    sample = self.datasets[dataset_idx][sample_idx_fallback]
        else:
            # Simple integer index - use random sampling to match original behavior
            sample_idx = random.randint(0, self.lengths[dataset_idx] - 1)
            try:
                sample = self.datasets[dataset_idx][sample_idx]
            except Exception as e:
                print(f"Warning: Failed to get sample {sample_idx} from dataset {self.dataset_names[dataset_idx]}: {e}")
                # Try another random sample
                sample_idx_retry = random.randint(0, self.lengths[dataset_idx] - 1)
                sample = self.datasets[dataset_idx][sample_idx_retry]
        
        # Add metadata about which dataset this came from
        if isinstance(sample, dict):
            sample['dataset'] = self.dataset_names[dataset_idx]
            sample['dataset_idx'] = dataset_idx
        
        return sample
    
    def set_epoch(self, epoch):
        """Set epoch for distributed training - propagate to all datasets"""
        self.epoch = epoch
        for dataset in self.datasets:
            if hasattr(dataset, 'set_epoch'):
                dataset.set_epoch(epoch)
    
    def get_dataset_stats(self):
        """Return statistics about dataset sampling"""
        return {
            'num_datasets': self.num_datasets,
            'dataset_names': self.dataset_names,
            'weights': self.weights,
            'lengths': self.lengths,
            'total_length': self.total_length
        }


class TaskWeightedConcatDataset(Dataset):
    """
    A dataset that combines multiple WeightedConcatDatasets (one per task)
    and samples from tasks based on task weights.
    
    This is useful when you have multiple tasks (e.g., depth, normals, segmentation)
    and want to control sampling at both task and dataset levels.
    
    Args:
        task_datasets (dict): Dict of {task_name: WeightedConcatDataset}
        task_weights (dict): Dict of {task_name: weight}
    """
    
    def __init__(self, task_datasets, task_weights):
        self.task_names = list(task_datasets.keys())
        self.task_datasets = task_datasets
        
        # Normalize task weights
        total_weight = sum(task_weights.values())
        self.task_weights = [task_weights[name] / total_weight for name in self.task_names]
        
        # Total length is sum of all task dataset lengths
        self.total_length = sum(len(ds) for ds in task_datasets.values())
        
        self.epoch = 0
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        """Sample a task based on weights, then sample from that task's dataset"""
        # Sample which task to use
        task_idx = random.choices(
            range(len(self.task_names)),
            weights=self.task_weights,
            k=1
        )[0]
        
        task_name = self.task_names[task_idx]
        
        # Get sample from that task's dataset
        sample = self.task_datasets[task_name][idx]
        
        # Add task metadata
        if isinstance(sample, dict):
            sample['task'] = task_name
        
        return sample
    
    def set_epoch(self, epoch):
        """Set epoch for distributed training - propagate to all task datasets"""
        self.epoch = epoch
        for task_dataset in self.task_datasets.values():
            if hasattr(task_dataset, 'set_epoch'):
                task_dataset.set_epoch(epoch)

