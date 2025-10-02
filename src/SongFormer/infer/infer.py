import argparse
import importlib
import json
import math
import multiprocessing as mp
import os
import time
from argparse import Namespace
from pathlib import Path

# monkey patch to fix issues in msaf
import scipy
import numpy as np

scipy.inf = np.inf

import librosa
import torch
from ema_pytorch import EMA
from loguru import logger
from muq import MuQ
from musicfm.model.musicfm_25hz import MusicFM25Hz
from omegaconf import OmegaConf
from tqdm import tqdm

mp.set_start_method("spawn", force=True)

MUSICFM_HOME_PATH = os.path.join("ckpts", "MusicFM")

BEFORE_DOWNSAMPLING_FRAME_RATES = 25
AFTER_DOWNSAMPLING_FRAME_RATES = 8.333

DATASET_LABEL = "SongForm-HX-8Class"
DATASET_IDS = [5]

TIME_DUR = 420
INPUT_SAMPLING_RATE = 24000

from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID
from postprocessing.functional import postprocess_functional_structure


def get_processed_ids(output_path):
    """Get already processed IDs from output directory"""
    ids = os.listdir(output_path)
    ret = []
    for x in ids:
        if x.endswith(".json"):
            ret.append(x.replace(".json", ""))
    return set(ret)


def get_processing_ids(input_path, processed_ids_set):
    """Get IDs to be processed from input directory"""
    ret = []
    with open(input_path) as f:
        for line in f:
            if line.strip() and Path(line.strip()).stem not in processed_ids_set:
                ret.append(line.strip())
    return ret


def load_checkpoint(checkpoint_path, device=None):
    """Load checkpoint from path"""
    if device is None:
        device = "cpu"

    if checkpoint_path.endswith(".pt"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
    elif checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint = {"model_ema": load_file(checkpoint_path, device=device)}
    else:
        raise ValueError("Unsupported checkpoint format. Use .pt or .safetensors")
    return checkpoint


def rule_post_processing(msa_list):
    if len(msa_list) <= 2:
        return msa_list

    result = msa_list.copy()

    while len(result) > 2:
        first_duration = result[1][0] - result[0][0]
        if first_duration < 1.0 and len(result) > 2:
            result[0] = (result[0][0], result[1][1])
            result = [result[0]] + result[2:]
        else:
            break

    while len(result) > 2:
        last_label_duration = result[-1][0] - result[-2][0]
        if last_label_duration < 1.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    while len(result) > 2:
        if result[0][1] == result[1][1] and result[1][0] <= 10.0:
            result = [(result[0][0], result[0][1])] + result[2:]
        else:
            break

    while len(result) > 2:
        last_duration = result[-1][0] - result[-2][0]
        if result[-2][1] == result[-3][1] and last_duration <= 10.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    return result


def inference(rank, queue_input: mp.Queue, queue_output: mp.Queue, args):
    """Run inference on the input audio"""
    device = f"cuda:{rank}"

    # MuQ model loading (this will automatically fetch the checkpoint from huggingface)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq = muq.to(device).eval()

    # MusicFM model loading
    musicfm = MusicFM25Hz(
        is_flash=False,
        stat_path=os.path.join(MUSICFM_HOME_PATH, "msd_stats.json"),
        model_path=os.path.join(MUSICFM_HOME_PATH, "pretrained_msd.pt"),
    )
    musicfm = musicfm.to(device)
    musicfm.eval()

    # Custom model loading based on the config
    module = importlib.import_module("models." + str(args.model))
    Model = getattr(module, "Model")
    hp = OmegaConf.load(os.path.join("configs", args.config_path))
    model = Model(hp)

    ckpt = load_checkpoint(checkpoint_path=os.path.join("ckpts", args.checkpoint))
    if ckpt.get("model_ema", None) is not None:
        logger.info("Loading EMA model parameters")
        model_ema = EMA(model, include_online_model=False)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(model_ema.ema_model.state_dict())
    else:
        logger.info("No EMA model parameters found, using original model")
        model.load_state_dict(ckpt["model"])

    model.to(device)
    model.eval()

    num_classes = args.num_classes
    dataset_id2label_mask = {}

    for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
        dataset_id2label_mask[key] = np.ones(args.num_classes, dtype=bool)
        dataset_id2label_mask[key][allowed_ids] = False

    with torch.no_grad():
        while True:
            item = queue_input.get()
            if not item:
                queue_output.put(None)
                break

            try:
                # Loading the audio file
                wav, sr = librosa.load(item, sr=INPUT_SAMPLING_RATE)
                audio = torch.tensor(wav).to(device)

                win_size = args.win_size
                hop_size = args.hop_size
                total_len = (
                    (audio.shape[0] // INPUT_SAMPLING_RATE) // TIME_DUR
                ) * TIME_DUR + TIME_DUR
                total_frames = math.ceil(total_len * AFTER_DOWNSAMPLING_FRAME_RATES)

                logits = {
                    "function_logits": np.zeros([total_frames, num_classes]),
                    "boundary_logits": np.zeros([total_frames]),
                }
                logits_num = {
                    "function_logits": np.zeros([total_frames, num_classes]),
                    "boundary_logits": np.zeros([total_frames]),
                }

                lens = 0
                i = 0
                while True:
                    start_idx = i * INPUT_SAMPLING_RATE
                    end_idx = min((i + win_size) * INPUT_SAMPLING_RATE, audio.shape[-1])
                    if start_idx >= audio.shape[-1]:
                        break
                    if end_idx - start_idx <= 1024:
                        continue
                    audio_seg = audio[start_idx:end_idx]

                    # MuQ embedding
                    muq_output = muq(audio_seg.unsqueeze(0), output_hidden_states=True)
                    muq_embd_420s = muq_output["hidden_states"][10]
                    del muq_output
                    torch.cuda.empty_cache()

                    # MusicFM embedding
                    _, musicfm_hidden_states = musicfm.get_predictions(
                        audio_seg.unsqueeze(0)
                    )
                    musicfm_embd_420s = musicfm_hidden_states[10]
                    del musicfm_hidden_states
                    torch.cuda.empty_cache()

                    wraped_muq_embd_30s = []
                    wraped_musicfm_embd_30s = []

                    for idx_30s in range(i, i + hop_size, 30):
                        start_idx_30s = idx_30s * INPUT_SAMPLING_RATE
                        end_idx_30s = min(
                            (idx_30s + 30) * INPUT_SAMPLING_RATE,
                            audio.shape[-1],
                            (i + hop_size) * INPUT_SAMPLING_RATE,
                        )
                        if start_idx_30s >= audio.shape[-1]:
                            break
                        if end_idx_30s - start_idx_30s <= 1024:
                            continue
                        wraped_muq_embd_30s.append(
                            muq(
                                audio[start_idx_30s:end_idx_30s].unsqueeze(0),
                                output_hidden_states=True,
                            )["hidden_states"][10]
                        )
                        torch.cuda.empty_cache()
                        wraped_musicfm_embd_30s.append(
                            musicfm.get_predictions(
                                audio[start_idx_30s:end_idx_30s].unsqueeze(0)
                            )[1][10]
                        )
                        torch.cuda.empty_cache()

                    wraped_muq_embd_30s = torch.concatenate(wraped_muq_embd_30s, dim=1)
                    wraped_musicfm_embd_30s = torch.concatenate(
                        wraped_musicfm_embd_30s, dim=1
                    )
                    all_embds = [
                        wraped_musicfm_embd_30s,
                        wraped_muq_embd_30s,
                        musicfm_embd_420s,
                        muq_embd_420s,
                    ]

                    if len(all_embds) > 1:
                        embd_lens = [x.shape[1] for x in all_embds]
                        max_embd_len = max(embd_lens)
                        min_embd_len = min(embd_lens)
                        if abs(max_embd_len - min_embd_len) > 4:
                            raise ValueError(
                                f"Embedding shapes differ too much: {max_embd_len} vs {min_embd_len}"
                            )

                        for idx in range(len(all_embds)):
                            all_embds[idx] = all_embds[idx][:, :min_embd_len, :]

                    embd = torch.concatenate(all_embds, axis=-1)

                    dataset_label = DATASET_LABEL
                    dataset_ids = torch.Tensor(DATASET_IDS).to(device, dtype=torch.long)
                    msa_info, chunk_logits = model.infer(
                        input_embeddings=embd,
                        dataset_ids=dataset_ids,
                        label_id_masks=torch.Tensor(
                            dataset_id2label_mask[
                                DATASET_LABEL_TO_DATASET_ID[dataset_label]
                            ]
                        )
                        .to(device, dtype=bool)
                        .unsqueeze(0)
                        .unsqueeze(0),
                        with_logits=True,
                    )

                    start_frame = int(i * AFTER_DOWNSAMPLING_FRAME_RATES)
                    end_frame = start_frame + min(
                        math.ceil(hop_size * AFTER_DOWNSAMPLING_FRAME_RATES),
                        chunk_logits["boundary_logits"][0].shape[0],
                    )

                    logits["function_logits"][start_frame:end_frame, :] += (
                        chunk_logits["function_logits"][0].detach().cpu().numpy()
                    )
                    logits["boundary_logits"][start_frame:end_frame] = (
                        chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                    )
                    logits_num["function_logits"][start_frame:end_frame, :] += 1
                    logits_num["boundary_logits"][start_frame:end_frame] += 1
                    lens += end_frame - start_frame

                    i += hop_size
                logits["function_logits"] /= logits_num["function_logits"]
                logits["boundary_logits"] /= logits_num["boundary_logits"]

                logits["function_logits"] = torch.from_numpy(
                    logits["function_logits"][:lens]
                ).unsqueeze(0)
                logits["boundary_logits"] = torch.from_numpy(
                    logits["boundary_logits"][:lens]
                ).unsqueeze(0)

                msa_infer_output = postprocess_functional_structure(logits, hp)

                assert msa_infer_output[-1][-1] == "end"
                if not args.no_rule_post_processing:
                    msa_infer_output = rule_post_processing(msa_infer_output)
                msa_json = []
                for idx in range(len(msa_infer_output) - 1):
                    msa_json.append(
                        {
                            "label": msa_infer_output[idx][1],
                            "start": msa_infer_output[idx][0],
                            "end": msa_infer_output[idx + 1][0],
                        }
                    )
                json.dump(
                    msa_json,
                    open(os.path.join(args.output_dir, f"{Path(item).stem}.json"), "w"),
                    indent=4,
                    ensure_ascii=False,
                )

                queue_output.put(None)

            except Exception as e:
                queue_output.put(None)
                logger.error(f"process {rank} error\n{item}\n{e}")


def deal_with_output(output_path, queue_output, length):
    """Handle output data from the queue"""
    pbar = tqdm(range(length), desc="getting inference output")
    for _ in pbar:
        data = queue_output.get()
        if not data:
            continue


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    gpu_num = args.gpu_num
    num_thread_per_gpu = args.num_thread_per_gpu
    debug = args.debug

    os.makedirs(output_path, exist_ok=True)

    processed_ids = get_processed_ids(output_path=output_path)
    processing_ids = get_processing_ids(input_path, processed_ids)

    num_threads = num_thread_per_gpu * gpu_num

    queue_input: mp.Queue = mp.Queue()
    queue_output: mp.Queue = mp.Queue()

    init_args = Namespace(
        output_dir=output_path,
        win_size=420,
        hop_size=420,
        num_classes=128,
        model=args.model,
        checkpoint=args.checkpoint,
        config_path=args.config_path,
        no_rule_post_processing=args.no_rule_post_processing,
    )

    processes = []

    if debug:
        queue_input.put(processing_ids[0])
        queue_input.put(None)

        inference(0, queue_input, queue_output, init_args)

        print("debug exit")
        exit(0)

    for thread_num in range(num_threads):
        rank = thread_num % gpu_num
        print(f"num_threads: {thread_num} on GPU {rank}")
        time.sleep(0.2)
        p = mp.Process(
            target=inference,
            args=(rank, queue_input, queue_output, init_args),
            daemon=True,
        )
        p.start()
        processes.append(p)

    for wav_id in tqdm(processing_ids, desc="add data to queue"):
        queue_input.put(wav_id)

    for _ in range(num_threads):
        queue_input.put(None)

    deal_with_output(output_path, queue_output, len(processing_ids))

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", "-i", type=str, required=True, help="Input file path"
    )
    parser.add_argument(
        "--output_path", "-o", type=str, required=True, help="Output file path"
    )
    parser.add_argument(
        "--gpu_num", "-gn", type=int, default=1, help="Number of GPUs, default is 1"
    )
    parser.add_argument(
        "--num_thread_per_gpu",
        "-tn",
        type=int,
        default=1,
        help="Number of threads per GPU, default is 1",
    )
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path")
    parser.add_argument("--config_path", type=str, help="Configuration file path")
    parser.add_argument(
        "--no_rule_post_processing",
        action="store_true",
        help="Disable rule-based post-processing",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    main(args=args)
