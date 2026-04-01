#!/usr/bin/env python3
"""
Merge Pi3 and MoGe v2 checkpoints to create the pretrained checkpoint for DAGE.
Usage:
    python scripts/merge_checkpoints.py 
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

import argparse
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file
import yaml

from dage.models.dage import DAGE



def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint from a local path or Hugging Face repo."""
    checkpoint_source = checkpoint_path
    local_checkpoint_path = Path(checkpoint_path)

    if Path(checkpoint_path).exists():
        if checkpoint_path.endswith(".safetensors"):
            checkpoint = safe_load_file(checkpoint_path)
        else:
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
            except:
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu"
                )
    else:
        candidate_filenames = [
            "model.pt",
            "model.safetensors",
            "pytorch_model.bin",
            "checkpoint.pt",
        ]

        cached_checkpoint_path = None
        last_error = None
        for filename in candidate_filenames:
            try:
                cached_checkpoint_path = hf_hub_download(
                    repo_id=checkpoint_source,
                    repo_type="model",
                    filename=filename,
                )
                break
            except Exception as error:
                last_error = error

        if cached_checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find a checkpoint file in Hugging Face repo "
                f"{checkpoint_source}. Tried: {candidate_filenames}"
            ) from last_error

        if cached_checkpoint_path.endswith(".safetensors"):
            checkpoint = safe_load_file(cached_checkpoint_path)
        else:
            try:
                checkpoint = torch.load(
                    cached_checkpoint_path, map_location="cpu", weights_only=True
                )
            except:
                checkpoint = torch.load(cached_checkpoint_path, map_location="cpu")
    
    return checkpoint


def get_state_dict(checkpoint: dict) -> dict:
    """Extract state dict from checkpoint, handling different key conventions."""
    if "ema_model" in checkpoint:
        return checkpoint["ema_model"]
    elif "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Merge Pi3 and MoGe v2 checkpoints to create DAGE pretrained checkpoint"
    )
    parser.add_argument(
        "--pi3-ckpt",
        default="yyfz233/Pi3",
        help="Path to trained Pi3  checkpoint . "
    )
    parser.add_argument(
        "--mogev2-ckpt",
        default="Ruicheng/moge-2-vitl",
        help="Path to pre-trained MoGe v2 checkpoint",
    )
    parser.add_argument(
        "--output",
        default="checkpoints/merged_pi3_mogev2.pt",
        help="Output path for the merged checkpoint.",
    )
    args = parser.parse_args()


    model_config = yaml.load(open('configs/model_config_dage.yaml', 'r'), Loader=yaml.FullLoader)["model"]["config"]

    dage_model = DAGE.from_pretrained(
        pretrained_model_name_or_path=args.pi3_ckpt,
        model_config=model_config,
        strict=False
    )
    dage_model_state_dict = dage_model.state_dict()

    new_state_dict = dict()

    # -------------------------------------------------------------------------
    # Step 1: Load MoGe v2 checkpoint — provides pre-trained HR encoder, neck, points head, mask head
    # -------------------------------------------------------------------------
    mogev2_ckpt = load_checkpoint(args.mogev2_ckpt)
    mogev2_state_dict = get_state_dict(mogev2_ckpt)

    count_mogev2 = 0
    for k, v in mogev2_state_dict.items():
        if k.startswith("encoder."):
            new_k = "hr_encoder." + k.replace("encoder.", "")
            if new_k in dage_model_state_dict.keys() and dage_model_state_dict[new_k].shape == v.shape:
                new_state_dict[new_k] = v
                print(f"Adding {new_k} to new state dict")
                count_mogev2 += 1
        elif k.startswith("neck."):
            new_k = "hr_neck." + k.replace("neck.", "")
            if new_k in dage_model_state_dict.keys() and dage_model_state_dict[new_k].shape == v.shape:
                new_state_dict[new_k] = v
                print(f"Adding {new_k} to new state dict")
                count_mogev2 += 1
        elif k.startswith("points_head."):
            new_k = "hr_points_head." + k.replace("points_head.", "")
            if new_k in dage_model_state_dict.keys() and dage_model_state_dict[new_k].shape == v.shape:
                new_state_dict[new_k] = v
                print(f"Adding {new_k} to new state dict")
                count_mogev2 += 1
        elif k.startswith("mask_head."):
            new_k = "hr_mask_head." + k.replace("mask_head.", "")
            if new_k in dage_model_state_dict.keys() and dage_model_state_dict[new_k].shape == v.shape:
                new_state_dict[new_k] = v
                print(f"Adding {new_k} to new state dict")
                count_mogev2 += 1
        elif k.startswith("scale_head."):
            new_k = "hr_scale_head." + k.replace("scale_head.", "")
            if new_k in dage_model_state_dict.keys() and dage_model_state_dict[new_k].shape == v.shape:
                new_state_dict[new_k] = v
                print(f"Adding {new_k} to new state dict")
                count_mogev2 += 1

    print("--------------------------------------------------------")
    print(f"  -> Copied {count_mogev2} keys from MoGe v2 checkpoint")
    print("--------------------------------------------------------")

    # -------------------------------------------------------------------------
    # Step 2: Load Pi3 Teacher checkpoint — provides pre-trained LR encoder
    # -------------------------------------------------------------------------
    pi3_ckpt = load_checkpoint(args.pi3_ckpt)
    pi3_state_dict = get_state_dict(pi3_ckpt)

    count_pi3 = 0
    for k, v in pi3_state_dict.items():
        if k in dage_model_state_dict.keys() and dage_model_state_dict[k].shape == v.shape:
            new_state_dict[k] = v
            print(f"Adding {k} to new state dict")
            count_pi3 += 1

    print("--------------------------------------------------------")
    print(f"  -> Copied {count_pi3} keys from Pi3 Teacher checkpoint")
    print("--------------------------------------------------------")

    print(f"Total keys in new state dict: {len(new_state_dict)}")
    print(f"total keys in dage model state dict: {len(dage_model_state_dict)}")

    print("Keys in dage model state dict but not in new state dict:")
    for k in dage_model_state_dict.keys():
        if k not in new_state_dict.keys():
            print(k)

    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint = {
        "model": new_state_dict,
        "config": model_config,
    }

    torch.save(save_checkpoint, str(output_path))

    print(f"\nMerged checkpoint saved to: {output_path}")


if __name__ == "__main__":
    main()
