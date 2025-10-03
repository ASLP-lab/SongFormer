import argparse
import importlib
import json
import os
from collections import defaultdict

import hydra
import numpy as np
import torch
from dataset.label2id import (
    DATASET_ID_ALLOWED_LABEL_IDS,
    DATASET_LABEL_TO_DATASET_ID,
)
from dataset.msa_info_utils import dump_msa_infos, load_msa_info
from omegaconf import OmegaConf
from postprocessing.functional import postprocess_functional_structure
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.results_visual import gen_visible_report, visualisation
import pdb

# from ema import LitEma
from ema_pytorch import EMA
from loguru import logger

def prefix_dict(d, prefix: str):
    if prefix:
        return d
    return {prefix + key: value for key, value in d.items()}


def vis_infer_chunk(
    device,
    model,
    embedding_dir,
    segments_dir,
    eval_id_scp_path,
    visual_id_list_path,
    dataset_label,
    dataset_ids,
    hparams,
    output_dir,
):
    model.eval()

    assert isinstance(dataset_ids, int) or isinstance(dataset_ids, str)

    dataset_ids = [int(dataset_ids)]

    frame_rates = hparams.frame_rates
    num_classes = hparams.num_classes
    time_dur = hparams.slice_dur

    data_id2store_name = defaultdict(list)
    assert len(embedding_dir.split()) >= 2, "Embedding dir should contain two paths"
    embedding_names = os.listdir(embedding_dir.split()[0])

    for x in embedding_names:
        tmp = "_".join(x.split(".")[0].split("_")[:-1])
        data_id2store_name[tmp].append(x)

    sorted_data_id2store_name = {
        k: sorted(v, key=lambda x: int(x.split(".")[0].split("_")[-1]))
        for k, v in data_id2store_name.items()
    }

    with open(eval_id_scp_path) as f:
        eval_lists = []
        for line in f:
            eval_lists.append(line.strip())

    eval_lists = frozenset(eval_lists)

    visual_id_lists = []
    with open(visual_id_list_path, "r") as f:
        data = json.load(f)
        for key, val in data.items():
            visual_id_lists.extend(val)
    visual_id_lists = frozenset(visual_id_lists)

    dataset_id2label_mask = {}
    for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
        dataset_id2label_mask[key] = np.ones(num_classes, dtype=bool)
        dataset_id2label_mask[key][allowed_ids] = False

    run_count = 0
    results = []
    with torch.no_grad():
        for data_id in tqdm(sorted_data_id2store_name):
            if data_id not in eval_lists:
                continue
            print(f"Evaluating {data_id}: {sorted_data_id2store_name[data_id]}")
            try:
                TIME_DUR = time_dur
                total_len = (
                    int(
                        sorted_data_id2store_name[data_id][-1]
                        .split(".")[0]
                        .split("_")[-1]
                    )
                    + TIME_DUR
                )
                logits = {
                    "function_logits": np.zeros(
                        [int(total_len * frame_rates), num_classes]
                    ),
                    "boundary_logits": np.zeros([int(total_len * frame_rates)]),
                }
                logits_num = {
                    "function_logits": np.zeros(
                        [int(total_len * frame_rates), num_classes]
                    ),
                    "boundary_logits": np.zeros([int(total_len * frame_rates)]),
                }
                if sorted_data_id2store_name is None:
                    continue

                lens = 0
                for numpy_file in sorted_data_id2store_name[data_id]:
                    start_time = int(numpy_file.split(".")[0].split("_")[-1])

                    # !!!!!!!!!!!!!!!!!!!!
                    musicfm_dir, muq_dir = embedding_dir.split()
                    musicfm_embedding = np.load(os.path.join(musicfm_dir, numpy_file))
                    muq_embedding = np.load(os.path.join(muq_dir, numpy_file))

                    # ============== along dim

                    embedding = np.concatenate(
                        [musicfm_embedding, muq_embedding], axis=-1
                    )

                    
                    embedding = torch.from_numpy(embedding).to(device)

                    start_frame = int(start_time * frame_rates)
                    end_frame = start_frame + min(
                        int(TIME_DUR * frame_rates), embedding.shape[1]
                    )


                    dataset_ids = torch.Tensor(dataset_ids).to(device, dtype=torch.long)

                    msa_info, chunk_logits = model.infer(
                        input_embeddings=embedding,
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

                    if (
                        chunk_logits["function_logits"][0]
                        .detach()
                        .cpu()
                        .numpy()
                        .shape[0]
                        != logits["function_logits"][start_frame:end_frame, :].shape[0]
                    ):

                        end_frame = (
                            start_frame
                            + chunk_logits["function_logits"][0]
                            .detach()
                            .cpu()
                            .numpy()
                            .shape[0]
                        )
                        print("-", end="")


                    logits["function_logits"][start_frame:end_frame, :] += (
                        chunk_logits["function_logits"][0].detach().cpu().numpy()
                    )
                    logits["boundary_logits"][start_frame:end_frame] = (
                        chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                    )
                    logits_num["function_logits"][start_frame:end_frame, :] += 1
                    logits_num["boundary_logits"][start_frame:end_frame] += 1
                    lens += end_frame - start_frame
                logits["function_logits"] /= logits_num["function_logits"]
                logits["boundary_logits"] /= logits_num["boundary_logits"]

                logits["function_logits"] = torch.from_numpy(
                    logits["function_logits"][:lens]
                ).unsqueeze(0)
                logits["boundary_logits"] = torch.from_numpy(
                    logits["boundary_logits"][:lens]
                ).unsqueeze(0)

                msa_infer_output = postprocess_functional_structure(logits, hparams)
                msa_pred_str = dump_msa_infos(msa_infer_output)
                msa_gt = load_msa_info(os.path.join(segments_dir, data_id + ".txt"))
                msa_gt_str = dump_msa_infos(msa_gt)

                # 可视化
                if data_id in visual_id_lists:

                    os.makedirs(f"{output_dir}/visualisation", exist_ok=True)
                    visualisation(
                        logits=logits,
                        msa_infer_output=msa_infer_output,
                        msa_gt=msa_gt,
                        data_id=data_id,
                        label_num=8,  # FIX
                        frame_rates=frame_rates,
                        output_path=f"{output_dir}/visualisation/vis_{data_id}.pdf",
                    )

                with open(os.path.join(output_dir, data_id + ".txt"), "w") as f_out:
                    f_out.write(msa_pred_str)
                results.extend(
                    [
                        "=" * 40,
                        data_id,
                        "=" * 40 + "\n pred:",
                        msa_pred_str,
                        "-" * 16 + "\n gt:",
                        msa_gt_str,
                    ]
                )
                run_count += 1

            except Exception as e:

                print(e)
                continue
    gen_visible_report(
        f"{output_dir}/visualisation/",
        f"{output_dir}/visualisation/reporting.pdf",
    )

    with open(os.path.join(output_dir, "result_chunk.txt"), "w") as f:
        f.write("\n".join(results))
