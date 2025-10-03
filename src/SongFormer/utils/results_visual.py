import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from dataset.label2id import ID_TO_LABEL
from pathlib import Path
from PyPDF2 import PdfMerger


def visualisation(
    logits, msa_infer_output, msa_gt, data_id, label_num, frame_rates, output_path
):
    assert output_path.endswith(".pdf")
    function_vals = (
        logits["function_logits"].squeeze().cpu().numpy()
    )  # [T, num_classes]
    boundary_vals = logits["boundary_logits"].squeeze().cpu().numpy()  # [T]

    top_classes = np.argsort(function_vals.mean(axis=0))[-label_num:]  # Top 7 by mean
    T = function_vals.shape[0]
    time_axis = np.arange(T) / frame_rates  # Convert to seconds

    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # ---- Function logits ----
    for cls in top_classes:
        ax[1].plot(time_axis, function_vals[:, cls], label=f"{ID_TO_LABEL[cls]}")

    ax[1].set_title("Top 7 Function logits by mean activation")
    ax[1].set_xlabel("Time (seconds)")
    ax[1].set_ylabel("Logit")
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(20))  # 每 10 秒一个主刻度
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(5))  # 每 1 秒一个次刻度
    ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))  # 一位小数

    ax[1].legend()
    ax[1].grid(True)

    # ---- Boundary logits ----
    ax[0].plot(time_axis, boundary_vals, label="Boundary logit")
    ax[0].set_title("Boundary logits")
    ax[0].set_ylabel("Logit")
    ax[0].legend()
    ax[0].grid(True)

    for t_sec, label in msa_infer_output:
        for a in ax:
            a.axvline(x=t_sec, color="red", linestyle="--", linewidth=0.8)
        ax[1].text(
            t_sec + 0.3,
            ax[1].get_ylim()[1] * 0.85,
            label,
            rotation=90,
            fontsize=8,
            color="red",
        )

    for t_sec, label in msa_gt:
        for a in ax:
            a.axvline(x=t_sec, color="blue", linestyle=":", linewidth=0.8)
        ax[1].text(
            t_sec + 0.3,
            ax[1].get_ylim()[1] * 0.6,
            label,
            rotation=90,
            fontsize=8,
            color="blue",
        )

    # 保存图像
    plt.suptitle(f"{data_id} — MSA Logits Overview", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path),
        bbox_inches="tight",
    )
    plt.close()


def gen_visible_report(pdf_dir, output_path):
    merger = PdfMerger()

    pdf_files = sorted(
        [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.endswith(".pdf") and Path(output_path).stem not in f
        ]
    )
    for pdf in pdf_files:
        merger.append(pdf)

    merger.write(output_path)
    merger.close()
