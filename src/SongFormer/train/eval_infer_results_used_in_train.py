import argparse
import os
from collections import defaultdict
from pathlib import Path
import mir_eval
import numpy as np
import pandas as pd
from dataset.custom_types import MsaInfo
from dataset.label2id import LABEL_TO_ID
from dataset.msa_info_utils import (
    load_msa_info,
)
from msaf.eval import compute_results
from postprocessing.calc_acc import cal_acc
from postprocessing.calc_iou import cal_iou
from tqdm import tqdm
from loguru import logger

# EVAL_LISTS_FILE_PATH = "/home/node59_tmpdata3/cbhao/msa/msa/src/dataset/separated_ids/harmonixset_separated_ids/eval.txt"
# EVAL_LISTS_FILE_PATH = "/home/node59_tmpdata3/cbhao/msa/msa/src/dataset/separated_ids/internal_data_sofa_clean/eval.txt"
# 合法的label
LEGAL_LABELS = {
    "end",
    "intro",
    "verse",
    "chorus",
    "bridge",
    "inst",
    "outro",
    "silence",
    "pre-chorus",
}


def to_inters_labels(msa_info: MsaInfo):
    label_ids = np.array([LABEL_TO_ID[x[1]] for x in msa_info[:-1]])
    times = [x[0] for x in msa_info]
    start_times = np.column_stack([np.array(times[:-1]), np.array(times[1:])])
    return start_times, label_ids


def eval_infer_results(
    ann_dir,
    est_dir,
    output_dir,
    eval_lists_file_path,
    prechorus2what=None,
):
    """
    Evaluate inference results against annotations.
    Args:
        ann_dir (str): Directory containing annotation files.
        est_dir (str): Directory containing estimated files.
        output_dir (str): Directory to save evaluation results.
        eval_lists_file_path (str): Path to the file containing evaluation lists.
        prechorus2what (str, optional): If specified, convert 'pre-chorus' to this label.
            Options are 'verse' or 'chorus'. Defaults to None.
    """
    os.makedirs(output_dir, exist_ok=True)

    ann_id_lists = os.listdir(ann_dir)
    ann_id_lists = [x for x in ann_id_lists if x.endswith(".txt")]
    est_id_lists = os.listdir(est_dir)
    est_id_lists = [x for x in est_id_lists if x.endswith(".txt")]

    eval_lists = []
    with open(eval_lists_file_path) as f:
        for line in f:
            eval_lists.append(line.strip() + ".txt")
    common_id_lists = set(ann_id_lists) & set(est_id_lists) & set(eval_lists)
    common_id_lists = list(common_id_lists)
    # pdb.set_trace()
    print(f"common number is {len(common_id_lists)}")

    resultes = []
    ious = {}
    for id in tqdm(common_id_lists):
        try:
            ann_msa = load_msa_info(os.path.join(ann_dir, id))
            est_msa = load_msa_info(os.path.join(est_dir, id))

            # assert all([x[1] in LEGAL_LABELS for x in ann_msa])
            # assert all([x[1] in LEGAL_LABELS for x in est_msa])

            if prechorus2what == None:
                pass
            elif prechorus2what == "verse":
                ann_msa = [
                    (t, "verse") if l == "pre-chorus" else (t, l) for t, l in ann_msa
                ]
                est_msa = [
                    (t, "verse") if l == "pre-chorus" else (t, l) for t, l in est_msa
                ]
            elif prechorus2what == "chorus":
                ann_msa = [
                    (t, "chorus") if l == "pre-chorus" else (t, l) for t, l in ann_msa
                ]
                est_msa = [
                    (t, "chorus") if l == "pre-chorus" else (t, l) for t, l in est_msa
                ]
            else:
                raise ValueError(f"{prechorus2what} is not supported")
            # print(id)
            ann_inter, ann_labels = to_inters_labels(ann_msa)
            est_inter, est_labels = to_inters_labels(est_msa)

            result = compute_results(
                ann_inter,
                est_inter,
                ann_labels,
                est_labels,
                bins=11,
                est_file="test.txt",
                weight=0.58,
            )
            acc = cal_acc(
                ann_msa,
                est_msa,
                post_digit=3,
            )
            # 结构为：[{label: str, iou: float, intsec_dur: float, uni_dur: float}, ...]
            ious[id] = cal_iou(
                ann_info=ann_msa,
                est_info=est_msa,
            )
            result["HitRate_1P"], result["HitRate_1R"], result["HitRate_1F"] = (
                mir_eval.segment.detection(ann_inter, est_inter, window=1, trim=False)
            )
            # pdb.set_trace()
            result.update({"id": Path(id).stem})
            result.update({"acc": acc})
            for v in ious[id]:
                result.update({f"iou-{v['label']}": v["iou"]})
            del result["track_id"]
            del result["ds_name"]

            resultes.append(result)
        except:
            # raise
            logger.error(f"{id} error")
            continue

    df = pd.DataFrame(resultes)
    df.to_csv(f"{output_dir}/eval_infer.csv", index=False)
    # import pdb

    # pdb.set_trace()
    # with open(f"{output_dir}/eval_infer.txt", "w") as f:
    #     print(f"- mean acc is {df['acc'].mean()}", file=f)
    #     print(f"- mean HR.5F is {df['HitRate_0.5F'].mean()}", file=f)
    #     print(f"- mean PWF is {df['PWF'].mean()}", file=f)
    #     print(f"- mean Sf is {df['Sf'].mean()}", file=f)

    intsec_dur_total = defaultdict(float)
    uni_dur_total = defaultdict(float)

    # pdb.set_trace()
    for tid, value in ious.items():
        for item in value:
            label = item["label"]
            intsec_dur_total[label] += item.get("intsec_dur", 0)
            uni_dur_total[label] += item.get("uni_dur", 0)

    total_intsec = sum(intsec_dur_total.values())
    total_uni = sum(uni_dur_total.values())
    overall_iou = total_intsec / total_uni if total_uni > 0 else 0.0

    class_ious = {}
    for label in intsec_dur_total:
        intsec = intsec_dur_total[label]
        uni = uni_dur_total[label]
        class_ious[label] = intsec / uni if uni > 0 else 0.0
    # import pdb

    # pdb.set_trace()
    summary = pd.DataFrame(
        [
            {
                "num_samples": len(df),
                "HR.5F": df["HitRate_0.5F"].mean(),
                "HR3F": df["HitRate_3F"].mean(),
                "HR1F": df["HitRate_1F"].mean(),
                "PWF": df["PWF"].mean(),
                "Sf": df["Sf"].mean(),
                "acc": df["acc"].mean(),
                "iou": overall_iou,
                **{f"iou_{k}": v for k, v in class_ious.items()},
            }
        ]
    )
    with open(f"{output_dir}/eval_infer_summary.md", "w") as f:
        print(summary.to_markdown(), file=f)
    # pdb.set_trace()
    summary.to_csv(f"{output_dir}/eval_infer_summary.csv", index=False)
    logger.info(f"Results saved to {output_dir}")

    return summary
