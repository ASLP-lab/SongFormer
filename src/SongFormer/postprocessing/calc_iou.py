import os
from dataset.custom_types import MsaInfo
from dataset.label2id import LABEL_TO_ID
from pprint import pprint


def load_msa_info(msa_info_path):
    msa_info: MsaInfo = []
    with open(msa_info_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            time_, label = line.split()
            time_ = float(time_)
            label = str(label)
            assert label in LABEL_TO_ID or label == "end", f"{label} not in LABEL_TO_ID"
            msa_info.append((time_, label))
    assert msa_info[-1][1] == "end", f"last {msa_info[-1][1]} != end"
    return msa_info


def msa_info_to_segments(msa_info):
    # skip the last "end"
    segments = []
    for i in range(len(msa_info) - 1):
        start = msa_info[i][0]
        end = msa_info[i + 1][0]
        label = msa_info[i][1]
        segments.append((start, end, label))
    return segments


def compute_iou_for_label(segments_a, segments_b, label):
    # segments_a, segments_b: [(start, end, label)]
    # only process the current label
    intervals_a = [(s, e) for s, e, l in segments_a if l == label]
    intervals_b = [(s, e) for s, e, l in segments_b if l == label]
    # sum up all intersections between a and b
    intersection = 0.0
    for sa, ea in intervals_a:
        for sb, eb in intervals_b:
            left = max(sa, sb)
            right = min(ea, eb)
            if left < right:
                intersection += right - left
    # union = total length of both sets - overlapping intersection
    length_a = sum([e - s for s, e in intervals_a])
    length_b = sum([e - s for s, e in intervals_b])
    union = length_a + length_b - intersection
    if union == 0:
        return 0.0
    return intersection / union, intersection, union


def compute_mean_iou(segments_a, segments_b, labels):
    ious = []
    for label in labels:
        iou, intsec_dur, uni_dur = compute_iou_for_label(segments_a, segments_b, label)
        ious.append(
            {"label": label, "iou": iou, "intsec_dur": intsec_dur, "uni_dur": uni_dur}
        )
    return ious


def cal_iou(ann_info, est_info):
    if type(ann_info) is str:
        assert os.path.exists(ann_info), f"{ann_info} not exists"
        ann_info = load_msa_info(ann_info)

    if type(est_info) is str:
        assert os.path.exists(est_info), f"{est_info} not exists"
        est_info = load_msa_info(est_info)

    segments_ann = msa_info_to_segments(ann_info)
    segments_est = msa_info_to_segments(est_info)

    occurred_labels = list(
        set([l for s, e, l in segments_ann]) | set(l for s, e, l in segments_est)
    )

    mean_iou = compute_mean_iou(segments_ann, segments_est, occurred_labels)
    return mean_iou


if __name__ == "__main__":
    ann_info = ""
    est_info = ""
    pprint(cal_iou(ann_info, est_info))
