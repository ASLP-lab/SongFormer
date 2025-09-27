import torch
import copy
from typing import List, Dict, Any


def average_checkpoints(checkpoint_paths: List[str], output_path: str = None):
    """
    Average the model and model_ema weights from multiple checkpoints

    Parameters:
    checkpoint_paths: List of checkpoint file paths
    output_path: Output path; if None, return the averaged checkpoint dictionary

    Returns:
    Averaged checkpoint dictionary
    """
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required")

    # Load the first checkpoint as the base
    print(f"Loading base checkpoint: {checkpoint_paths[0]}")
    avg_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")

    if len(checkpoint_paths) == 1:
        if output_path:
            torch.save(avg_checkpoint, output_path)
        return avg_checkpoint

    # Initialize accumulators
    avg_model_state = copy.deepcopy(avg_checkpoint["model"])
    avg_model_ema_state = None

    if "model_ema" in avg_checkpoint:
        avg_model_ema_state = copy.deepcopy(avg_checkpoint["model_ema"])

    # Accumulate the weights from the other checkpoints
    for i, ckpt_path in enumerate(checkpoint_paths[1:], 1):
        print(f"Processing checkpoint {i + 1}/{len(checkpoint_paths)}: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Accumulate model weights
        for key in avg_model_state.keys():
            if key in ckpt["model"]:
                avg_model_state[key] += ckpt["model"][key]

        # Accumulate model_ema weights (if available)
        if avg_model_ema_state is not None and "model_ema" in ckpt:
            for key in avg_model_ema_state.keys():
                if key in ckpt["model_ema"]:
                    avg_model_ema_state[key] += ckpt["model_ema"][key]

    # Compute the average
    num_checkpoints = len(checkpoint_paths)
    print(f"Averaging over {num_checkpoints} checkpoints...")

    for key in avg_model_state.keys():
        avg_model_state[key] = avg_model_state[key] / num_checkpoints

    if avg_model_ema_state is not None:
        for key in avg_model_ema_state.keys():
            avg_model_ema_state[key] = avg_model_ema_state[key] / num_checkpoints

    # Update the checkpoint dictionary
    avg_checkpoint["model"] = avg_model_state
    if avg_model_ema_state is not None:
        avg_checkpoint["model_ema"] = avg_model_ema_state

    # Save (if an output path is specified)
    if output_path:
        print(f"Saving averaged checkpoint to: {output_path}")
        torch.save(avg_checkpoint, output_path)

    return avg_checkpoint


def average_checkpoints_memory_efficient(
    checkpoint_paths: List[str], output_path: str = None
):
    """
    Memory efficient version: Load and process checkpoints one by one, suitable for large models
    """
    if not checkpoint_paths:
        raise ValueError("At least one checkpoint path is required")

    print(f"Loading base checkpoint: {checkpoint_paths[0]}")
    avg_checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")

    if len(checkpoint_paths) == 1:
        if output_path:
            torch.save(avg_checkpoint, output_path)
        return avg_checkpoint

    # Convert to float32 for better precision
    for key in avg_checkpoint["model"].keys():
        avg_checkpoint["model"][key] = avg_checkpoint["model"][key].float()

    if "model_ema" in avg_checkpoint:
        for key in avg_checkpoint["model_ema"].keys():
            avg_checkpoint["model_ema"][key] = avg_checkpoint["model_ema"][key].float()

    # Load and accumulate checkpoints one by one
    for i, ckpt_path in enumerate(checkpoint_paths[1:], 1):
        print(f"Processing checkpoint {i + 1}/{len(checkpoint_paths)}: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Accumulate model weights
        for key in avg_checkpoint["model"].keys():
            if key in ckpt["model"]:
                avg_checkpoint["model"][key] += ckpt["model"][key].float()

        # Accumulate model_ema weights
        if "model_ema" in avg_checkpoint and "model_ema" in ckpt:
            for key in avg_checkpoint["model_ema"].keys():
                if key in ckpt["model_ema"]:
                    avg_checkpoint["model_ema"][key] += ckpt["model_ema"][key].float()

        # Free memory
        del ckpt
        torch.cuda.empty_cache()

    # Compute the average
    num_checkpoints = len(checkpoint_paths)
    print(f"Averaging over {num_checkpoints} checkpoints...")

    for key in avg_checkpoint["model"].keys():
        avg_checkpoint["model"][key] /= num_checkpoints

    if "model_ema" in avg_checkpoint:
        for key in avg_checkpoint["model_ema"].keys():
            avg_checkpoint["model_ema"][key] /= num_checkpoints

    if output_path:
        print(f"Saving averaged checkpoint to: {output_path}")
        torch.save(avg_checkpoint, output_path)

    return avg_checkpoint


# Example usage
if __name__ == "__main__":
    # Method 1: Simple usage
    checkpoint_paths = []

    # Average and save
    average_checkpoints(checkpoint_paths, "")

    # Method 2: Get the averaged checkpoint and further process it
    # avg_ckpt = average_checkpoints(checkpoint_paths)
    # print("Averaged checkpoint keys:", avg_ckpt.keys())

    # Method 3: Use memory-efficient version (suitable for large models)
    # average_checkpoints_memory_efficient(checkpoint_paths, 'averaged_checkpoint_efficient.pt')
