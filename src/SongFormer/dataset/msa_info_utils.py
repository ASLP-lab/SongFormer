from dataset.custom_types import MsaInfo
from dataset.label2id import LABEL_TO_ID


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


def load_msa_infos(msa_str):
    msa_info: MsaInfo = []
    for line in msa_str:
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


def dump_msa_info(msa_info_path, msa_info: MsaInfo):
    with open(msa_info_path, "w") as f:
        for time_, label in msa_info:
            f.write(f"{time_} {label}\n")


def dump_msa_infos(msa_info: MsaInfo):
    mas_strs = []
    for time_, label in msa_info:
        mas_strs.append(f"{round(time_, 2)} {label}")

    return "\n".join(mas_strs)
