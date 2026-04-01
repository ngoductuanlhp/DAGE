import os
import io
import numpy as np
import itertools

def blockwise_shuffle(x, rng, block_shuffle):
    if block_shuffle is None:
        return rng.permutation(x).tolist()
    else:
        assert block_shuffle > 0
        blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
        shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
        shuffled_list = [item for block in shuffled_blocks for item in block]
        return shuffled_list


def get_seq_from_start_id(num_views, id_ref, ids_all, rng, min_interval=1, max_interval=25, video_prob=0.5, fix_interval_prob=0.5, block_shuffle=None):
    """
    Sophisticated sampling strategy for frame sequences.
    
    Args:
        num_views: number of views to return
        id_ref: the reference id (first id)  
        ids_all: all the ids
        rng: random number generator
    Returns:
        pos: list of positions of the views in ids_all, i.e., index for ids_all
        is_video: True if the views are consecutive
    """
    # assert self.min_interval > 0, f"min_interval should be > 0, got {self.min_interval}"
    # assert self.min_interval <= self.max_interval, f"min_interval should be <= max_interval, got {self.min_interval} and {self.max_interval}"
    # assert id_ref in ids_all

    # min_interval = self.stride_range[0]
    # max_interval = self.stride_range[1]
    assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
    assert min_interval <= max_interval, f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
    assert id_ref in ids_all
    
    # pos_ref = ids_all.index(id_ref)
    pos_ref = id_ref
    all_possible_pos = np.arange(pos_ref, len(ids_all))
    remaining_sum = len(ids_all) - 1 - pos_ref

    if remaining_sum >= num_views - 1:
        if remaining_sum == num_views - 1:
            assert ids_all[-num_views] == id_ref
            return True, [pos_ref + i for i in range(num_views)], True
            
        max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
        intervals = [
            rng.choice(range(min_interval, max_interval + 1))
            for _ in range(num_views - 1)
        ]

        # if video or collection
        if rng.random() < video_prob:
            # if fixed interval or random
            if rng.random() < fix_interval_prob:
                # regular interval
                fixed_interval = rng.choice(
                    range(
                        1,
                        min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                    )
                )
                intervals = [fixed_interval for _ in range(num_views - 1)]
            is_video = True
        else:
            is_video = False

        pos = list(itertools.accumulate([pos_ref] + intervals))
        pos = [p for p in pos if p < len(ids_all)]
        pos_candidates = [p for p in all_possible_pos if p not in pos]
        
        if len(pos) < num_views and len(pos_candidates) > 0:
            additional_needed = num_views - len(pos)
            if len(pos_candidates) >= additional_needed:
                additional_pos = rng.choice(pos_candidates, additional_needed, replace=False).tolist()
            else:
                additional_pos = pos_candidates
            pos.extend(additional_pos)

        pos = (
            sorted(pos)
            if is_video
            else blockwise_shuffle(pos, rng, block_shuffle)
        )
    else:
        return False, [], False
        # # Handle case when there aren't enough frames
        # if not self.allow_repeat:
        #     # Just use all available frames
        #     pos = list(range(pos_ref, len(ids_all)))
        #     # Pad with the last frame if needed
        #     while len(pos) < num_views:
        #         pos.append(len(ids_all) - 1)
        #     is_video = True
        # else:
        #     # Use the repeat logic from reference implementation
        #     uniq_num = remaining_sum + 1  # +1 to include the reference frame
        #     new_pos_ref = rng.choice(np.arange(pos_ref + 1)) if pos_ref > 0 else 0
        #     new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            
        #     if new_remaining_sum > 0 and uniq_num > 1:
        #         new_max_interval = min(self.max_interval, new_remaining_sum // (uniq_num - 1))
        #         new_intervals = [
        #             rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
        #         ]
        #     else:
        #         new_intervals = []

        #     revisit_random = rng.random()
        #     video_random = rng.random()

        #     if rng.random() < self.fix_interval_prob and video_random < self.video_prob:
        #         # regular interval
        #         if new_remaining_sum > 0 and uniq_num > 1:
        #             new_max_interval = min(self.max_interval, new_remaining_sum // (uniq_num - 1))
        #             fixed_interval = rng.choice(range(1, new_max_interval + 1))
        #             new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            
        #     pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

        #     is_video = False
        #     if revisit_random < 0.5 or self.video_prob == 1.0:  # revisit, video / collection
        #         is_video = video_random < self.video_prob
        #         pos = (
        #             self.blockwise_shuffle(pos, rng, self.block_shuffle)
        #             if not is_video
        #             else pos
        #         )
        #         num_full_repeat = num_views // uniq_num
        #         pos = (
        #             pos * num_full_repeat
        #             + pos[: num_views - len(pos) * num_full_repeat]
        #         )
        #     elif revisit_random < 0.9:  # random
        #         pos = rng.choice(pos, num_views, replace=True).tolist()
        #     else:  # ordered
        #         pos = sorted(rng.choice(pos, num_views, replace=True)).tolist()

    # Ensure we have exactly num_views frames
    if len(pos) > num_views:
        pos = pos[:num_views]
    elif len(pos) < num_views:
        # Pad with last frame
        last_pos = pos[-1] if pos else 0
        pos.extend([last_pos] * (num_views - len(pos)))
        
    assert len(pos) == num_views
    return True, pos, is_video