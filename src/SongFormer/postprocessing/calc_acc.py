import os
import bisect
from dataset.msa_info_utils import (
    load_msa_info,
)
from dataset.custom_types import MsaInfo
import glob
import pdb
import pandas as pd


def cal_acc(ann_info: MsaInfo | str, est_info: MsaInfo | str, post_digit: int = 3):
    if type(ann_info) is str:
        assert os.path.exists(ann_info), f"{ann_info} not exists"
        ann_info = load_msa_info(ann_info)

    if type(ann_info) is str:
        assert os.path.exists(est_info), f"{est_info} not exists"
        est_info = load_msa_info(est_info)

    ann_info_time = [
        int(round(time_, post_digit) * (10**post_digit)) for time_, label in ann_info
    ]
    est_info_time = [
        int(round(time_, post_digit) * (10**post_digit)) for time_, label in est_info
    ]

    common_start_time = max(ann_info_time[0], est_info_time[0])
    common_end_time = min(ann_info_time[-1], est_info_time[-1])

    time_points = set()
    time_points.add(common_start_time)
    time_points.add(common_end_time)

    for time_ in ann_info_time:
        if time_ >= common_start_time and time_ <= common_end_time:
            time_points.add(time_)
    for time_ in est_info_time:
        if time_ >= common_start_time and time_ <= common_end_time:
            time_points.add(time_)

    time_points = sorted(list(time_points))
    total_duration = 0
    total_score = 0

    for idx in range(len(time_points) - 1):
        duration = time_points[idx + 1] - time_points[idx]
        ann_label = ann_info[bisect.bisect_right(ann_info_time, time_points[idx]) - 1][
            1
        ]
        est_label = est_info[bisect.bisect_right(est_info_time, time_points[idx]) - 1][
            1
        ]
        total_duration += duration
        if ann_label == est_label:
            total_score += duration
    return total_score / total_duration


if __name__ == "__main__":
    ext_paths = glob.glob("")
    results = []
    for ext_path in ext_paths:
        try:
            ann_path = os.path.join(
                "",
                os.path.basename(ext_path).split(".")[0] + ".txt",
            )
            results.append(
                {
                    "data_id": os.path.basename(ext_path).split(".")[0],
                    "acc": cal_acc(
                        ann_info=ann_path,
                        est_info=ext_path,
                    ),
                }
            )
        except Exception as e:
            print(e)
            continue
    df = pd.DataFrame(results)
    print(df["acc"].mean())
