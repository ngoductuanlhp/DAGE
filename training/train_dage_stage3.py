# Training code for MoGe temporal model fine-tuning
  
import logging
import math
import os
import shutil
import random
from omegaconf import OmegaConf
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import accelerate
import datasets
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from packaging import version
from tqdm.auto import tqdm
from collections import defaultdict
from einops import rearrange


try:
    from diffusers.utils.torch_utils import is_compiled_module
except ImportError:
    # Fallback if diffusers is not available
    def is_compiled_module(module):
        return hasattr(module, "_orig_mod")

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

from training.dataloaders.unified_video_dataloader import UnifiedVideoDataLoader
from training.loss import Pi3Loss, Pi3DistillLoss
from training.util.lr_scheduler import IterStep, IterExponential, IterCosine
import time
import utils3d


# from third_party.moge.moge.model.moge_model_v2_pose_v2 import MoGeModelV2PoseV2


# Add evaluation imports
import h5py
from decord import VideoReader, cpu
from evaluation.geocrafter.metrics import compute_metrics
from evaluation.moge.utils.tools import key_average
from evaluation.geocrafter.config import BENCHMARK_CONFIGS
from tabulate import tabulate
import json
import open3d as o3d
from torch.utils.data._utils.collate import default_collate

from dage.models.dage import DAGE

from third_party.pi3.models.pi3_teacher import Pi3Teacher

from training.util.misc import move_to_device


from evaluation.mv_recon.data_pi3 import NRGBD
from evaluation.mv_recon.utils import accuracy, completion, umeyama
from evaluation.geocrafter.metrics import compute_metrics
from evaluation.geocrafter.config import BENCHMARK_CONFIGS

try:
    import wandb
    _wandb_available = True
except ImportError:
    _wandb_available = False

def is_wandb_available():
    return _wandb_available

logger = get_logger(__name__, log_level="INFO")


def create_param_groups_with_different_lr(model, base_lr, head_blocks_lr_multiplier=1.0, other_params_lr_multiplier=0.1, weight_decay=0.01):
    """
    Create parameter groups with different learning rates for different model components.
    Also handles weight decay separately for bias parameters.
    
    Args:
        model: The model to extract parameters from
        base_lr: Base learning rate
        head_blocks_lr_multiplier: Multiplier for default learning rate (default: 1.0)
        other_params_lr_multiplier: Multiplier for reduced learning rate (default: 0.1)
        weight_decay: Weight decay value (default: 0.01)
    
    Returns:
        tuple: (param_groups, learnable_params, param_info)
    """
    default_lr_params = []  # 1.0x LR
    reduced_10x_params = []  # 0.1x LR
    reduced_50x_params = []  # 0.02x LR
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         if not "camera" in name:# NOTE for pose model, we don't use temporal blocks
    #             head_blocks_params.append((name, param))
    #         else:
    #             other_learnable_params.append((name, param))

    # Group 1: Default LR (1.0x)
    default_lr_prefixes = [
        "adapter.",
        "hr_points_head.",
        "hr_mask_head.",
        "hr_neck.",

        "hr_scale_head.",
        "cls_decoder.",
    ]
    
    # Group 2: Reduced by 10x (0.1x)
    reduced_10x_prefixes = [
        "camera_decoder.",
        "camera_head.",
        "decoder.",
        "token_splitter.",
    ]
    
    # Group 3: Reduced by 50x (0.02x)
    reduced_50x_prefixes = [
        "none"
    ]

    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(name.startswith(prefix) for prefix in default_lr_prefixes):
                # Group 1: Default LR
                default_lr_params.append((name, param))
            elif any(name.startswith(prefix) for prefix in reduced_10x_prefixes):
                # Group 2: LR divided by 10
                reduced_10x_params.append((name, param))
            elif any(name.startswith(prefix) for prefix in reduced_50x_prefixes):
                # Group 3: LR divided by 50
                reduced_50x_params.append((name, param))
            else:
                # Default behavior: put in default LR group
                default_lr_params.append((name, param))
    
    def handle_weight_decay(params, weight_decay, lr, group_name):
        """
        Handle weight decay for parameters, setting weight_decay to 0.0 for bias parameters.
        
        Args:
            params: List of (name, param) tuples
            weight_decay: Weight decay value
            lr: Learning rate
            group_name: Name prefix for the parameter groups
        
        Returns:
            List of parameter group dictionaries
        """
        decay = []
        no_decay = []
        for name, param in params:
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        groups = []
        if no_decay:
            groups.append({
                "params": no_decay, 
                "weight_decay": 0.0, 
                'lr': lr,
                # 'name': f'{group_name}_no_decay'
            })
        if decay:
            groups.append({
                "params": decay, 
                "weight_decay": weight_decay, 
                'lr': lr,
                # 'name': f'{group_name}_decay'
            })
        
        return groups
    
    # Create parameter groups with different learning rates and weight decay handling
    param_groups = []
    
    default_lr = base_lr * 1.0  # Default LR
    reduced_10x_lr = base_lr * 0.1  # LR divided by 10
    reduced_50x_lr = base_lr * 0.02  # LR divided by 50
    
    if default_lr_params:
        param_groups.extend(handle_weight_decay(default_lr_params, weight_decay, default_lr, 'default_lr'))
    
    if reduced_10x_params:
        param_groups.extend(handle_weight_decay(reduced_10x_params, weight_decay, reduced_10x_lr, 'reduced_10x'))
    
    if reduced_50x_params:
        param_groups.extend(handle_weight_decay(reduced_50x_params, weight_decay, reduced_50x_lr, 'reduced_50x'))
    
    # Collect all learnable params for gradient clipping
    learnable_params = [param for _, param in default_lr_params + reduced_10x_params + reduced_50x_params]
    
    # Parameter info for logging
    default_lr_decay_count = sum(1 for name, param in default_lr_params if param.requires_grad and param.ndim > 1 and not name.endswith(".bias"))
    default_lr_no_decay_count = sum(1 for name, param in default_lr_params if param.requires_grad and (param.ndim <= 1 or name.endswith(".bias")))
    reduced_10x_decay_count = sum(1 for name, param in reduced_10x_params if param.requires_grad and param.ndim > 1 and not name.endswith(".bias"))
    reduced_10x_no_decay_count = sum(1 for name, param in reduced_10x_params if param.requires_grad and (param.ndim <= 1 or name.endswith(".bias")))
    reduced_50x_decay_count = sum(1 for name, param in reduced_50x_params if param.requires_grad and param.ndim > 1 and not name.endswith(".bias"))
    reduced_50x_no_decay_count = sum(1 for name, param in reduced_50x_params if param.requires_grad and (param.ndim <= 1 or name.endswith(".bias")))
    
    param_info = {
        'default_lr_count': len(default_lr_params),
        'reduced_10x_count': len(reduced_10x_params),
        'reduced_50x_count': len(reduced_50x_params),
        'default_lr_decay_count': default_lr_decay_count,
        'default_lr_no_decay_count': default_lr_no_decay_count,
        'reduced_10x_decay_count': reduced_10x_decay_count,
        'reduced_10x_no_decay_count': reduced_10x_no_decay_count,
        'reduced_50x_decay_count': reduced_50x_decay_count,
        'reduced_50x_no_decay_count': reduced_50x_no_decay_count,
        'default_lr': default_lr,
        'reduced_10x_lr': reduced_10x_lr,
        'reduced_50x_lr': reduced_50x_lr,
        'weight_decay': weight_decay
    }
    
    return param_groups, learnable_params, param_info

def remove_previous_checkpoints(args, logger, prefix="checkpoint", protected_ckpt=True):
    if args.checkpoints_total_limit is not None:
        checkpoints = os.listdir(args.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1].split(".")[0]))
        
        # Function to check if a checkpoint should be protected
        def is_protected_checkpoint(checkpoint_num, interval=5000):
            """Protect checkpoint 1000 and every multiple of the interval"""
            return checkpoint_num == 1000 or checkpoint_num % interval == 0
        
        # Separate protected and removable checkpoints
        protected_checkpoints = []
        removable_checkpoints = []
        
        for checkpoint in checkpoints:
            checkpoint_num = int(checkpoint.split("-")[-1].split(".")[0])
            if is_protected_checkpoint(checkpoint_num) and not os.path.isdir(os.path.join(args.output_dir, checkpoint)) and protected_ckpt:
                protected_checkpoints.append(checkpoint)
            else:
                removable_checkpoints.append(checkpoint)
        
        # Only consider removable checkpoints for deletion
        # We need to account for protected checkpoints in our total count
        total_checkpoints = len(checkpoints)
        if total_checkpoints >= args.checkpoints_total_limit:
            # Calculate how many we need to remove from removable checkpoints
            num_to_remove = total_checkpoints - args.checkpoints_total_limit + 1
            # Only remove from removable checkpoints, up to the number available
            num_to_remove = min(num_to_remove, len(removable_checkpoints))
            removing_checkpoints = removable_checkpoints[0:num_to_remove]
            
            if removing_checkpoints:
                logger.info(
                    f"{total_checkpoints} checkpoints exist ({len(protected_checkpoints)} protected), removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                    try:
                        if os.path.isdir(removing_checkpoint):
                            shutil.rmtree(removing_checkpoint)
                        elif os.path.isfile(removing_checkpoint):
                            os.remove(removing_checkpoint)
                        else:
                            logger.warning(f"Checkpoint path {removing_checkpoint} does not exist or is neither file nor directory")
                    except OSError as e:
                        logger.error(f"Failed to remove checkpoint {removing_checkpoint}: {e}")
            else:
                logger.info(f"All {len(removable_checkpoints)} removable checkpoints are needed, keeping all checkpoints")


def load_config(config_path):
    """Load training configuration from YAML file using OmegaConf"""
    if not os.path.exists(config_path):
        raise ValueError(f"Config file {config_path} does not exist")
    
    # Load config using OmegaConf
    config = OmegaConf.load(config_path)
    
    # Calculate gradient_accumulation_steps based on GPU count if not specified
    if not hasattr(config, "gradient_accumulation_steps") or config.gradient_accumulation_steps is None:
        # Get total number of GPUs across all nodes (for distributed training)
        if "WORLD_SIZE" in os.environ:
            # When using torchrun, WORLD_SIZE = total number of processes across all nodes
            num_gpus = int(os.environ["WORLD_SIZE"])
        else:
            # Fallback to local GPU count for single-node training
            num_gpus = torch.cuda.device_count()
        
        # config.gradient_accumulation_steps = max(1, config.effective_batch_size // (config.batch_size_per_gpu * num_gpus))
        if hasattr(config, "use_dynamic_batch_size") and config.use_dynamic_batch_size:
            assert config.effective_batch_size is None, "effective_batch_size must be None when use_dynamic_batch_size is True"
            config.gradient_accumulation_steps = 1
        else:
            config.gradient_accumulation_steps = max(1, config.effective_batch_size // (config.batch_size_per_gpu * num_gpus))
        
        
    return config

def unwrap_model(model, accelerator):
    """Unwrap model from accelerator and compilation wrappers"""
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output

    return get_output_hook

def add_hook(net, mem, mapping_layers):
    for n, m in net.named_modules():
        if n in mapping_layers:
            m.register_forward_hook(get_activation(mem, n))

def evaluate_video_depth(model, device, logger):
    """
    Evaluate video depth on Sintel dataset.
    Only runs on main process.
    
    Args:
        model: The model to evaluate (unwrapped)
        device: The device to run evaluation on
        logger: Logger for logging results
        
    Returns:
        dict: Evaluation metrics (depth metrics from Sintel)
    """
    from decord import VideoReader, cpu
    
    logger.info("=" * 80)
    logger.info("Running Video Depth Evaluation on Sintel")
    logger.info("=" * 80)
    
    model.eval()
    
    # Get Sintel benchmark config
    if "sintel" not in BENCHMARK_CONFIGS:
        logger.warning("Sintel benchmark not found in BENCHMARK_CONFIGS")
        return None
    
    benchmark_config = BENCHMARK_CONFIGS["sintel"]
    benchmark_path = benchmark_config["path"]
    
    # Load samples from filename_list.txt
    meta_file_path = os.path.join(benchmark_path, 'filename_list.txt')
    if not os.path.exists(meta_file_path):
        logger.warning(f"Meta file not found: {meta_file_path}")
        return None
        
    samples = []
    with open(meta_file_path, "r") as f:
        for line in f.readlines():
            video_path, data_path = line.split()
            samples.append(dict(
                video_path=os.path.join(benchmark_path, video_path),
                data_path=os.path.join(benchmark_path, data_path)
            ))
    
    # Limit number of samples for faster training evaluation
    max_eval_samples = 5
    samples = samples[:max_eval_samples]
    
    metrics_list = []
    
    with torch.no_grad():
        for i, sample in tqdm(enumerate(samples), total=len(samples), desc="Evaluating video depth", leave=False):
            try:
                video_path = sample["video_path"]
                data_path = sample["data_path"]
                height = benchmark_config["height"]
                width = benchmark_config["width"]
                use_weight = benchmark_config["use_weight"]
                
                # Load ground truth
                with h5py.File(data_path, "r") as file:
                    gt_mask = file['valid_mask'][:].astype(np.bool_)  # T H W
                    gt_pmap = file['point_map'][:].astype(np.float32)  # T H W C
                gt_pmap = torch.from_numpy(gt_pmap).to(device).float()
                gt_mask = torch.from_numpy(gt_mask).to(device).bool()
                
                # Load video data
                vid = VideoReader(video_path, ctx=cpu(0))
                frames_idx = list(range(0, len(vid), 1))
                frames = vid.get_batch(frames_idx).asnumpy().astype(np.float32)
                
                frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).to(device) / 255.0  # T 3 H W
                
                
                # Prepare video tensor
                video_tensor = frames_tensor.unsqueeze(0)  # 1 T 3 H W
                
                # Run inference
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    output = model.infer(
                        video_tensor, 
                        lr_max_size=252,
                        hr_resolution_level=9
                    )
                
                pred = output['local_points']  # T H W C
                pred_mask = output['mask']
                
                
                # Compute metrics
                metrics, _, _ = compute_metrics(
                    pred,
                    gt_pmap,
                    gt_mask,
                    use_weight=use_weight,
                )
                
                metrics_list.append(metrics["points_affine_invariant"])
                
            except Exception as e:
                logger.warning(f"Error evaluating sample {i}: {e}")
                continue
    
    # Calculate averages
    eval_dict = None
    if metrics_list:
        # Average all metrics
        mean_metrics = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if values and isinstance(values[0], (int, float, np.number)):
                mean_metrics[key] = float(np.mean(values))
        
        logger.info(f"Sintel Video Depth Results (n={len(metrics_list)}): {mean_metrics}")
        logger.info("=" * 80)
        
        eval_dict = mean_metrics
    else:
        logger.warning("No valid samples evaluated")
    
    model.train()
    return eval_dict


def main():
    if len(sys.argv) < 2:
        print("Error: Please provide a YAML config file path")
        print("Usage: python train.py config.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    args = load_config(config_path)    

    print(args)

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        # diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Set tasks
    # assert args.tasks, "No task is provided. Set --tasks."

    # Save training arguments in a .txt file
    if accelerator.is_main_process:
        args_dict = vars(args)
        args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
        args_str += f'\nnum_processes: {accelerator.num_processes}'
        args_str += f'\ndistributed_type: {accelerator.distributed_type}'
        args_path = os.path.join(args.output_dir, "arguments.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(args_path, 'w') as file:
            file.write(args_str)
        
        # Also save training arguments as a .yaml file
        config_to_save = OmegaConf.create(args)
        config_to_save.num_processes = accelerator.num_processes
        config_to_save.distributed_type = str(accelerator.distributed_type)
        
        yaml_args_path = os.path.join(args.output_dir, "training_config.yaml")
        with open(yaml_args_path, 'w') as file:
            OmegaConf.save(config_to_save, file)


    assert args.model.name == "DAGE", "Only DAGE is supported for this script"
    # NOTE dynamic model name based on args.model.name
    if args.pretrained_model_name_or_path is not None:
        pose_model = eval(args.model.name).from_pretrained(args.pretrained_model_name_or_path, strict=False, model_config=args.model.config)
    else:
        logger.info(f"Initializing model {args.model.name} with config: {args.model.config}")
        pose_model = eval(args.model.name)(**args.model.config)

    # Freeze all parameters except the temporal adaptation modules
    
    trainable_substrings = [
        "hr_mask_head.",
    ]
    # non_trainable_substrings = [
    #     "encoder.",
    # ]

    for name, param in pose_model.named_parameters():
        if any(name.lower().startswith(substr) for substr in trainable_substrings):
            param.requires_grad = True
        else:
            param.requires_grad = False

    pose_model.train()

    # Log how many parameters will be optimised
    num_trainable = sum(p.numel() for p in pose_model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in pose_model.parameters())
    logger.info(f"Pose model parameters: {num_trainable / 1e6:.2f}M trainable / {num_total / 1e6:.2f}M total")


    # Set up EMA model
    if accelerator.is_main_process:
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.999 * averaged_model_parameter + 0.001 * model_parameter
        ema_model = torch.optim.swa_utils.AveragedModel(pose_model, device=accelerator.device, avg_fn=ema_avg_fn)

    # Create parameter groups with different learning rates
    param_groups, learnable_params, param_info = create_param_groups_with_different_lr(
            pose_model, 
            args.learning_rate,
            head_blocks_lr_multiplier=1.0,
            other_params_lr_multiplier=0.1,
            weight_decay=args.adam_weight_decay
        )
    
    # Log parameter group information
    logger.info(f"Default LR parameters (1.0x): {param_info['default_lr_count']} (decay: {param_info['default_lr_decay_count']}, no_decay: {param_info['default_lr_no_decay_count']})")
    logger.info(f"Reduced 10x LR parameters (0.1x): {param_info['reduced_10x_count']} (decay: {param_info['reduced_10x_decay_count']}, no_decay: {param_info['reduced_10x_no_decay_count']})")
    logger.info(f"Reduced 50x LR parameters (0.02x): {param_info['reduced_50x_count']} (decay: {param_info['reduced_50x_decay_count']}, no_decay: {param_info['reduced_50x_no_decay_count']})")
    logger.info(f"Default LR: {param_info['default_lr']}")
    logger.info(f"Reduced 10x LR: {param_info['reduced_10x_lr']}")
    logger.info(f"Reduced 50x LR: {param_info['reduced_50x_lr']}")
    logger.info(f"Weight decay: {param_info['weight_decay']} (applied to non-bias parameters only)")
    
    optimizer = AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )



    lr_func = IterCosine(
        total_iter_length=args.lr_total_iter_length*accelerator.num_processes, 
        min_lr_ratio=1e-3,
        warmup_steps=args.lr_exp_warmup_steps*accelerator.num_processes,
    )
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)


    use_moge = getattr(args, "use_moge", False)
    use_dynamic_batch_size = getattr(args, "use_dynamic_batch_size", False)
    use_edge_loss = getattr(args, "use_edge_loss", False)
    use_real_loss = getattr(args, "use_real_loss", True)
    camera_loss_weight = getattr(args, "camera_loss_weight", 0.1)
    feat_distill_type = getattr(args, "feat_distill_type", 'mse')
    metric_scale_loss = getattr(args, "metric_scale_loss", False)

    
    train_dataloader = UnifiedVideoDataLoader(
        config=args.datasets,
        batch_size=args.batch_size_per_gpu,
        num_workers_per_dataset=args.dataloader_num_workers,
        shuffle=True,
        pin_memory=True,
        resolutions=getattr(args, "resolutions", None),
        frame_range=getattr(args, "frame_range", None),
        use_moge=use_moge,
        moge_augmentation=getattr(args, "moge_augmentation", None),
        process_index=accelerator.process_index,
        use_dynamic_batch_size=use_dynamic_batch_size,
    )

    # Prepare everything with `accelerator` (Move to GPU)
    pose_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pose_model, optimizer, train_dataloader, lr_scheduler
    )



    # Mixed precision and weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
        # pose_model.to(dtype=weight_dtype)
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
        # pose_model.to(dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # Initialize trackers on GPU 0 and GPU 1 to allow logging from both
    # (GPU 0 logs distill losses, GPU 1 logs non-distill losses)
    if accelerator.process_index == 0 or accelerator.process_index == 1:
        tracker_config = OmegaConf.to_container(args, resolve=True)
        tracker_config["tasks"] = "_".join(tracker_config["tasks"])
        tracker_config["model"] = "_".join(tracker_config["model"])
        if "resolutions" in tracker_config:
            # Convert resolution lists to strings like "512x384" and then join with underscores
            resolution_strings = [f"{res[0]}x{res[1]}" for res in tracker_config["resolutions"]]
            tracker_config["resolutions"] = "_".join(resolution_strings)
        if "frame_range" in tracker_config:
            tracker_config["frame_range"] = f"{tracker_config['frame_range'][0]}-{tracker_config['frame_range'][1]}"
        if "eval_benchmarks" in tracker_config:
            tracker_config["eval_benchmarks"] = "_".join(tracker_config["eval_benchmarks"])
        if "moge_augmentation" in tracker_config:
            tracker_config["moge_augmentation"] = "_".join(tracker_config["moge_augmentation"])
        # tracker_config["data"] = "_".join(tracker_config["data"])
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.batch_size_per_gpu * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0

    # Resume training from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:  
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))

            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            # Load EMA model
            if accelerator.is_main_process:
                ema_model_path = os.path.join(args.output_dir, f"checkpoint-ema-{global_step}.pt")
                ema_model_state = torch.load(ema_model_path)
                ema_model.module.load_state_dict(ema_model_state["ema_model"])
                # global_step = ema_model_state["global_step"]
                logger.info(f"Loaded EMA model weights from {ema_model_path}")
    else:
        initial_global_step = 0
        initial_global_epoch = 0

    # Progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps), 
        initial=initial_global_step, 
        desc="Steps", 
        disable=not accelerator.is_local_main_process,) 


    # NOTE reset seed to avoid training on the same data when resuming training
    if args.seed is not None:
        set_seed(args.seed + initial_global_step, device_specific=True)   


    ######################################

    loss_function = Pi3Loss(train_conf=True, use_edge_loss=False)

    train_loss = 0.0
    train_loss_dict = defaultdict(list)

    pose_model.train()

    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(pose_model):   
            batch = move_to_device(batch, accelerator.device)  
            rgb_images = batch["rgb"]
            original_height, original_width = rgb_images.shape[-2:]

            original_area = original_height * original_width
            original_aspect_ratio = original_width / original_height
            
            
            prior_max_size = random.choice(list[int](range(252, 336+1, 14)))
            if original_width > original_height:
                prior_width = prior_max_size
                prior_height = int((prior_width / original_aspect_ratio) // 14 * 14)
            else:
                prior_height = prior_max_size
                prior_width = int((prior_height * original_aspect_ratio) // 14 * 14)
            prior_resolution = (prior_height, prior_width)


            num_tokens = accelerate.utils.broadcast_object_list([random.randint(1200, 4000)])[0]
            
            with accelerator.autocast():
                pred_dict = pose_model(
                    rgb_images,
                    precomputed_hidden=None,
                    prior_resolution=prior_resolution,
                    num_tokens=num_tokens,
                    get_lr_points=False,
                )

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
                loss, loss_dict = loss_function(
                    pred=pred_dict,
                    gt_raw=batch,
                    device=accelerator.device,
                    use_moge=use_moge,
                    skip_cam_loss=True,
                    metric_scale_loss=False
                )
            
            
            avg_loss = accelerator.gather(loss.repeat(args.batch_size_per_gpu)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps
            
            
            for loss_name, loss_value in loss_dict.items():
                train_loss_dict[loss_name].append(loss_value.item() / args.gradient_accumulation_steps)

            # Backpropagate
            accelerator.backward(loss)
            
            # Clean up tensors after backward pass to free memory
            del pred_dict
            del batch
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(pose_model.parameters(), args.max_grad_norm)
                
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:

            # EMA update            
            if accelerator.is_main_process:
                ema_model.update_parameters(pose_model)

            progress_bar.update(1)
            global_step += 1

            if accelerator.process_index == 0:
                accelerator.log({"loss/train": train_loss}, step=global_step)
            
            # Log losses from both GPU 0 and GPU 1 independently to capture all metrics
            # Some losses only appear on GPU 0 (distill), some only on GPU 1 (non-distill)
            # Log from both processes with gpu-specific prefixes
            if accelerator.process_index == 0 or accelerator.process_index == 1:
                gpu_prefix = f"loss_gpu{accelerator.process_index}"
                for loss_name, loss_value_list in train_loss_dict.items():
                    loss_value = torch.tensor(loss_value_list).mean()
                    accelerator.log({f"{gpu_prefix}/{loss_name}": loss_value.item()}, step=global_step)
            
            # Log learning rates for both parameter groups
            current_lrs = lr_scheduler.get_last_lr()
            if accelerator.process_index == 0:
                if len(current_lrs) > 1:
                    accelerator.log({"lr/head_blocks": current_lrs[0]}, step=global_step)
                    accelerator.log({"lr/other_params": current_lrs[1]}, step=global_step)
                else:
                    accelerator.log({"lr": current_lrs[0]}, step=global_step)

            train_loss = 0.0
            train_loss_dict = defaultdict(list)
            
            if global_step % 100 == 0:  # Periodically clear cache
                torch.cuda.empty_cache()


            if global_step % args.checkpointing_steps == 0:
                accelerator.wait_for_everyone() # NOTE should or should not wait for everyone?
                logger.info(f"Entered Saving Code at global step {global_step} checkpointing_steps {args.checkpointing_steps}")
                if accelerator.is_main_process:
                    # Run video depth evaluation on Sintel
                    
                    try:
                        video_depth_eval_dict = evaluate_video_depth(
                            model=ema_model.module,
                                device=accelerator.device,
                                logger=logger
                            )
                        
                    
                        # Log video depth evaluation metrics to tracker
                        if video_depth_eval_dict is not None:
                            accelerator.log({
                                f"eval/video_depth_sintel_{key}": value for key, value in video_depth_eval_dict.items()
                            }, step=global_step)
                    except Exception as e:
                        logger.warning(f"Error evaluating video depth: {e}")
                        
                    
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    remove_previous_checkpoints(args, logger, prefix="checkpoint", protected_ckpt=False)
                    remove_previous_checkpoints(args, logger, prefix="checkpoint-ema", protected_ckpt=False)
                    
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    # Save EMA model
                    ema_model_save_path = os.path.join(args.output_dir, f"checkpoint-ema-{global_step}.pt")
                    torch.save(
                        {
                            "ema_model": ema_model.module.state_dict(),
                            "global_step": global_step,
                            "config": args.model.config,
                        }, 
                        ema_model_save_path
                    )
                    logger.info(f"Saved EMA model weights to {ema_model_save_path}")

                    model_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                    torch.save({
                        "model": unwrap_model(pose_model, accelerator).state_dict(),
                        "global_step": global_step,
                        "config": args.model.config,
                    }, model_save_path)
                    logger.info(f"Saved model weights to {model_save_path}")
                accelerator.wait_for_everyone() # NOTE should or should not wait for everyone?

        # Log loss and learning rate for progress bar
        current_lrs = lr_scheduler.get_last_lr()
        if len(current_lrs) > 1:
            logs = {"step_loss": loss.detach().item(), "lr_temporal": current_lrs[0], "lr_other": current_lrs[1]}
        else:
            logs = {"step_loss": loss.detach().item(), "lr": current_lrs[0]}
        progress_bar.set_postfix(**logs)

        # Break training
        if global_step >= args.max_train_steps:
            break     
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = unwrap_model(pose_model, accelerator)
        # Save the fine-tuned MoGe model weights for later reuse
        model_save_path = os.path.join(args.output_dir, "pose_model.pt")
        torch.save({
            "model": unwrapped_model.state_dict(),
            "config": args.model.config,
        }, model_save_path)
        logger.info(f"Saved MoGe model weights to {model_save_path}")

        # Save EMA model
        ema_model_save_path = os.path.join(args.output_dir, "ema_model.pt")
        torch.save(
            {
                "ema_model": ema_model.module.state_dict(),
                "config": args.model.config,
            }, 
            ema_model_save_path)
        logger.info(f"Saved EMA model weights to {ema_model_save_path}")
    
    logger.info(f"Finished training.")

    accelerator.end_training()

if __name__ == "__main__":
    main()