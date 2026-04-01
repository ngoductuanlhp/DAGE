import numpy as np
import torch
from accelerate import Accelerator
import torch.utils
from torch.utils.data import BatchSampler, Sampler, DistributedSampler
import torch.utils.data
from typing import Optional
import random


class DynamicCustomRandomSampler(Sampler):
    """Dynamic random sampling under a constraint: each sample in the batch has the same feature,
    which is chosen randomly from a known pool of 'features' for each batch.
    
    Unlike the original CustomRandomSampler, this version can be updated dynamically
    with new view_size and feat_idx parameters for each batch.
    
    The index returned is a tuple (sample_idx, view_idx, feat_idx).
    """

    def __init__(
        self,
        dataset,
        process_index=0,
    ):
            
        self.dataset = dataset
        self.process_index = process_index


        self.len_dataset = len(dataset)
        

        self.epoch = None
        
        # Dynamic parameters that can be updated
        self.current_view_size = None
        self.current_feat_idx = None

    def __len__(self):
        return self.len_dataset

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update_parameters(self, view_size, feat_idx):
        """Update dynamic parameters for the current batch."""
        self.current_view_size = view_size
        self.current_feat_idx = feat_idx

    def __iter__(self):
        if self.epoch is None:
            raise ValueError(
                "Epoch number not set. Please call 'set_epoch(epoch)' before iterating."
            )

        seed = self.epoch + 788 + self.process_index * 1000
        rng = np.random.default_rng(seed=seed)
        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.len_dataset)
        rng.shuffle(sample_idxs)

        # Yield indices with current dynamic parameters
        for idx in sample_idxs:
            if self.current_view_size is None or self.current_feat_idx is None:
                raise RuntimeError("Dynamic parameters not set. Call update_parameters() before iterating.")
            
            if self.current_feat_idx is None:
                yield (idx, self.current_view_size)
            else:
                yield (idx, self.current_view_size, self.current_feat_idx)


class DynamicBatchedSampler(Sampler):
    """Dynamic batch sampler that adjusts batch size based on total target views per GPU.
    
    This sampler ensures that the total number of views (batch_size * num_views) 
    remains approximately constant across different batches, even when num_views varies.
    """

    def __init__(
        self,
        sampler: DynamicCustomRandomSampler,
        target_total_views: int = 48,
        min_view_size: int = 4,
        max_view_size: int = 16,
        number_of_resolutions: Optional[int] = None,
        seed: int = 42,
        view_size_weights: Optional[dict] = None,
        force_divisible: bool = True,
    ):
        """
        Args:
            sampler: Instance of DynamicCustomRandomSampler
            target_total_views: Target total views per batch (batch_size * num_views)
            min_view_size: Minimum number of views per sample
            max_view_size: Maximum number of views per sample  
            number_of_resolutions: Number of resolution features (None if not used)
            seed: Random seed for reproducibility
            view_size_weights: Optional dict mapping view_size -> weight for sampling
        """
        self.sampler = sampler
        self.target_total_views = target_total_views
        self.min_view_size = min_view_size
        self.max_view_size = max_view_size
        self.number_of_resolutions = number_of_resolutions
        self.force_divisible = force_divisible
        self.seed = seed
        self.rng = random.Random()
        
        # Setup view size sampling
        if force_divisible:
            # self.possible_view_sizes = [v for v in range(min_view_size, max_view_size + 1) if (self.target_total_views // v) * v >= self.target_total_views * 0.90] # ensure that all gpus has at least 80% of the target total views
            self.possible_view_sizes = [v for v in range(min_view_size, max_view_size + 1) if (self.target_total_views // v) * v >= self.target_total_views * 0.5] # ensure that all gpus has at least 80% of the target total views

            # print(f"Possible view sizes: {self.possible_view_sizes}")
        else:
            self.possible_view_sizes = list(range(min_view_size, max_view_size + 1))
        
        if view_size_weights is None:
            # Uniform sampling by default
            self.view_size_weights = {size: 1.0 for size in self.possible_view_sizes}
        else:
            self.view_size_weights = view_size_weights
            
        # Normalize weights
        weights = [self.view_size_weights.get(size, 0.0) for size in self.possible_view_sizes]
        total_weight = sum(weights)
        self.normalized_weights = np.array(weights) / total_weight if total_weight > 0 else np.ones(len(weights)) / len(weights)
        
        self.epoch = 0

        # print(f"Possible view sizes: {self.possible_view_sizes}")
        # print(f"View size weights: {self.view_size_weights}")
        # print(f"Normalized weights: {self.normalized_weights}")

    def set_epoch(self, epoch):
        """Set epoch for both this sampler and the underlying sampler."""
        self.epoch = epoch
        self.sampler.set_epoch(epoch)
        self.rng.seed(epoch * 100 + self.seed)

    def __iter__(self):
        """Yield batches with dynamic batch sizes based on view count."""
        sampler_iterator = iter(self.sampler)
        
        while True:
            try:
                # Sample random view size for this batch
                random_view_size = int(np.random.choice(self.possible_view_sizes, p=self.normalized_weights))
                
                # Sample random feature index if using resolutions
                if self.number_of_resolutions is not None and self.number_of_resolutions > 0:
                    # if self.number_of_resolutions > 1:
                    #     # Bias towards lower resolution indices (like original code)
                    #     p = np.ones(self.number_of_resolutions)
                    #     p[:self.number_of_resolutions // 2] *= 2
                    #     p = p / p.sum()
                    #     random_feat_idx = int(np.random.choice(self.number_of_resolutions, p=p))
                    # else:
                    #     random_feat_idx = 0
                    random_feat_idx = int(np.random.choice(self.number_of_resolutions))
                else:
                    random_feat_idx = None
                
                # Calculate dynamic batch size to maintain target total views
                batch_size = max(1, self.target_total_views // random_view_size)
                
                # Update sampler with new parameters
                self.sampler.update_parameters(random_view_size, random_feat_idx)
                
                # Collect samples for current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)
                        current_batch.append(item)
                    except StopIteration:
                        break
                
                if not current_batch:
                    break  # No more samples
                    
                yield current_batch
                
            except StopIteration:
                break

    def __len__(self):
        # Return a reasonable estimate
        avg_view_size = (self.min_view_size + self.max_view_size) / 2
        avg_batch_size = max(1, self.target_total_views // avg_view_size)
        return len(self.sampler) // avg_batch_size


def make_dynamic_sampler(
    dataset,
    target_total_views: int = 48,
    number_of_resolutions: Optional[int] = None,
    min_num_frames: int = 4,
    max_num_frames: int = 16,
    shuffle: bool = True,
    view_size_weights: Optional[dict] = None,
    seed: int = 42,
    process_index: int = 0,
):
    """
    Create a dynamic sampler that adjusts batch size based on view count.
    
    Args:
        dataset: The dataset to sample from
        target_total_views: Target total views per batch (batch_size * num_views)
        number_of_resolutions: Number of resolution features (None if not used)
        min_num_frames: Minimum number of views per sample
        max_num_frames: Maximum number of views per sample
        shuffle: Whether to shuffle samples
        drop_last: Whether to drop last incomplete batch
        seed: Random seed
        view_size_weights: Optional weights for view size sampling
        num_replicas: Number of distributed replicas
        rank: Rank of current replica
        
    Returns:
        DynamicBatchedSampler instance
    """
    if not shuffle:
        raise NotImplementedError("Non-shuffled sampling not implemented yet")
    
    sampler = DynamicCustomRandomSampler(
        dataset=dataset,
        process_index=process_index,
    )
    
    return DynamicBatchedSampler(
        sampler=sampler,
        target_total_views=target_total_views,
        min_view_size=min_num_frames,
        max_view_size=max_num_frames,
        number_of_resolutions=number_of_resolutions,
        seed=seed,
        view_size_weights=view_size_weights,
    )


class DynamicDataLoaderWrapper:
    """
    Wrapper class to create DataLoader with dynamic batch sampling.
    Similar to the DynamicTorchDataset in the reference implementation.
    """
    
    def __init__(
        self,
        dataset,
        target_total_views: int = 48,
        number_of_resolutions: Optional[int] = None,
        min_num_frames: int = 4,
        max_num_frames: int = 16,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        collate_fn=None,
        worker_init_fn=None,
        persistent_workers: bool = False,
        seed: int = 42,
        view_size_weights: Optional[dict] = None,
    ):
        self.dataset = dataset
        self.target_total_views = target_total_views
        self.number_of_resolutions = number_of_resolutions
        self.min_num_frames = min_num_frames
        self.max_num_frames = max_num_frames
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.view_size_weights = view_size_weights
        
        # Create the dynamic batch sampler
        self.batch_sampler = make_dynamic_sampler(
            dataset=dataset,
            target_total_views=target_total_views,
            number_of_resolutions=number_of_resolutions,
            min_num_frames=min_num_frames,
            max_num_frames=max_num_frames,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            view_size_weights=view_size_weights,
        )
    
    def get_loader(self, epoch: int):
        """Create DataLoader for the given epoch."""
        print(f"Building dynamic dataloader with epoch: {epoch}")
        
        # Set epoch for the batch sampler
        self.batch_sampler.set_epoch(epoch)
        
        # Set epoch on dataset if it supports it
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)
        
        return torch.utils.data.DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=self.worker_init_fn,
        )


# if __name__ == "__main__":

#     batch_sampler = make_dynamic_sampler(
#         dataset=your_dataset,
#         target_total_views=48,
#         min_num_frames=4,
#         max_num_frames=16
#     )

#     dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
