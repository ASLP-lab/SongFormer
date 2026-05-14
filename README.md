<p align="center">
  <img src="figs/logo.png" width="50%" />
</p>


<h1 align="center">SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision</h1>

<div align="center">

<div style="text-align: center;">
  <img src="https://pfst.cf2.poecdn.net/base/image/3edb451ba5a7f21a2883ff2daef0815c5bd5c7551700754d7b1416894aa4213a?pmaid=486214493" alt="Python">  
  <img src="https://pfst.cf2.poecdn.net/base/image/7c8950244a7e0fa562345fadba0b80f7f2c516aa9b5b4b3440e0543b5ae357fe?pmaid=486214490" alt="License">  
  <a href="https://arxiv.org/abs/2510.02797">
    <img src="https://pfst.cf2.poecdn.net/base/image/cfca907eb1c2792af5128a680b48a47bdb0ceb1d1bdea83daff70d1cc34bb10b?pmaid=486214495" alt="arXiv Paper">
  </a>
  <a href="https://github.com/ASLP-lab/SongFormer">
    <img src="https://pfst.cf2.poecdn.net/base/image/7245776690a7a9de76e20f487a1002303b67d43ba6e815707ef5aa0ca460b8c3?pmaid=486214489" alt="GitHub">
  </a>
  <a href="https://huggingface.co/spaces/ASLP-lab/SongFormer">
    <img src="https://pfst.cf2.poecdn.net/base/image/f52d68641dacda8ce41acd49518acc697ea2f08070e6726aaa8160b48838c785?pmaid=486214498" alt="HuggingFace Space">
  </a>
  <a href="https://huggingface.co/ASLP-lab/SongFormer">
    <img src="https://pfst.cf2.poecdn.net/base/image/6211cc78bac4cc37aaa886fa6335032e7aedb5de092b32557b1b4324a1da4dd8?pmaid=486214494" alt="HuggingFace Model">
  </a>
  <a href="https://huggingface.co/datasets/ASLP-lab/SongFormDB">
    <img src="https://pfst.cf2.poecdn.net/base/image/b4b527bba20f316becb731bfa05c26502d726ab97121619afc4c7d359d3596e0?pmaid=486214496" alt="Dataset SongFormDB">
  </a>
  <a href="https://huggingface.co/datasets/ASLP-lab/SongFormBench">
    <img src="https://pfst.cf2.poecdn.net/base/image/21545a56eb4b744383845625bc365964ac071275f36bee985ba563fda55fe28f?pmaid=486214492" alt="Dataset SongFormBench">
  </a>
  <a href="https://discord.gg/p5uBryC4Zs">
    <img src="https://pfst.cf2.poecdn.net/base/image/f82a8fe87ee99b3aa4a5198683a37c0e44a62b8e9eca8a8f64b0590293a8cf12?pmaid=486214491" alt="Discord">
  </a>
  <a href="http://www.npu-aslp.org/">
    <img src="https://img.shields.io/badge/🏫-ASLP-grey?labelColor=lightgrey" alt="lab">
  </a>
</div>

</div>

<div align="center">
  <h3>
    Chunbo Hao<sup>1*</sup>, Ruibin Yuan<sup>2,6*</sup>, Jixun Yao<sup>1</sup>, Qixin Deng<sup>3,6</sup>,<br>Xinyi Bai<sup>4,6</sup>, Yanbo Wang<sup>5</sup>, Wei Xue<sup>2</sup>, Lei Xie<sup>1†</sup>
  </h3>


  <p>
    <sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>†</sup>Corresponding author
  </p>
  <p>
    <sup>1</sup>Audio, Speech and Language Processing Group (ASLP@NPU),<br>School of Computer Science, Northwestern Polytechnical University<br>
    <sup>2</sup>Hong Kong University of Science and Technology<br>
    <sup>3</sup>Northwestern University<br>
    <sup>4</sup>Cornell University<br>
    <sup>5</sup>University of New South Wales<br>
    <sup>6</sup>Multimodal Art Projection (M-A-P)
  </p>

</div>

----

[ English ｜ [中文](./README_ZH.md) ]

SongFormer is a music structure analysis framework that leverages multi-resolution self-supervised representations and heterogeneous supervision, accompanied by the large-scale multilingual dataset SongFormDB and the high-quality benchmark SongFormBench to foster fair and reproducible research.

![](figs/songformer.png)

## 📢 News and Updates

🔥 **October 3, 2025**  
**Open-sourced Training and Evaluation Code** – We have released the full training and evaluation code to support and promote community development and further research.

🔥 **October 2, 2025**  
**One-Click Inference on Hugging Face Launched** – Successfully deployed our one-click inference feature on the Hugging Face platform, making model testing and usage more accessible and user-friendly.

🔥 **September 30, 2025**  
**SongFormer Inference Package Released** – The complete SongFormer inference code and pre-trained checkpoint models are now publicly available for download and use.

🔥 **September 26, 2025**  
**SongFormDB and SongFormBench Launched** – We introduced our large-scale music dataset **SongFormDB** and comprehensive benchmark suite **SongFormBench**, both now available on Hugging Face to facilitate research and evaluation in Music structure analysis.

## 🚀 QuickStart

This model supports Hugging Face's from_pretrained method. To quickly get started with this code, you need to do two things:

1. Follow the instructions in `Setting up Python Environment` to configure your Python environment
2. Visit our [Hugging Face model page](https://huggingface.co/ASLP-lab/SongFormer), and run the code provided in the README

## 🌟 Key Highlights

We've achieved **breakthrough performance** in music structure analysis, setting new benchmarks across the board:

- ✨ **State-of-the-art accuracy** on both Western and Chinese music datasets
- ⚡ **Blazing fast inference** - faster than comparable models
- 💰 **Cost-effective** - No API fees, runs locally on single GPU

### ⏱️ Speed Comparison

**Process entire songs in just 2-4 seconds!** Here's how we stack up:

| Model                   | Processing Time | Note            |
| ----------------------- | --------------- | --------------- |
| **🏆 SongFormer (Ours)** | **2-4 seconds** |                 |
| LinkSeg-7Labels         | 3-5 seconds     |                 |
| All-In-One              | 9-12 seconds    |                 |
| SongPrep Fine-tuned     | 9-12 seconds    |                 |
| SongPrep End2End        | 22-26 seconds   | Contains lyrics |
| Gemini 2.5 Pro          | 30-90 seconds   | Contains lyrics |

*Benchmarked on NVIDIA L40 GPU (excluding model loading)*

### 📊 Performance Metrics

- **ACC**: Overall boundary detection accuracy
- **HR.5F**: Hit Rate with 0.5-second tolerance (fine-grained precision)
- **HR3F**: Hit Rate with 3-second tolerance

#### SongFormBench-HarmonixSet

| Method                   | ACC       | HR.5F     | HR3F      |
| ------------------------ | --------- | --------- | --------- |
| **Baseline Methods**     |           |           |           |
| Harmonic-CNN*            | 0.680     | 0.559     | —         |
| SpecTNT (24s)*           | 0.701     | 0.570     | —         |
| SpecTNT (36s)*           | 0.723     | 0.558     | —         |
| All-In-One               | 0.740     | 0.596     | 0.730     |
| MusicFM-Zhang et al.*    | 0.725     | 0.640     | 0.729     |
| MuQ_iter*                | 0.772     | —         | —         |
| LinkSeg-7Labels          | 0.780     | 0.630     | 0.762     |
| TA (Zhang et al., 2025)* | 0.787     | 0.610     | 0.801     |
| Gemini 2.5 Pro           | 0.748     | 0.423     | **0.813** |
| **SongFormer (Ours)**    |           |           |           |
| SongFormer (HX)          | 0.795     | **0.703** | 0.784     |
| SongFormer (HX+E+H)      | 0.806     | 0.697     | 0.780     |
| SongFormer (HX+E+H+G)    | **0.807** | 0.696     | 0.780     |

#### SongFormBench-CN

| Method                | ACC       | HR.5F     | HR3F      |
| --------------------- | --------- | --------- | --------- |
| **Baseline Methods**  |           |           |           |
| All-In-One            | 0.834     | 0.563     | 0.771     |
| LinkSeg-7Labels       | 0.828     | 0.518     | 0.757     |
| Gemini 2.5 Pro        | 0.806     | 0.412     | 0.833     |
| **SongFormer (Ours)** |           |           |           |
| SongFormer (HX)       | 0.848     | 0.675     | **0.856** |
| SongFormer (HX+E+H)   | 0.890     | **0.690** | 0.852     |
| SongFormer (HX+E+H+G) | **0.891** | 0.688     | 0.851     |


#### RWC-Pop

| Method                   | ACC       | HR.5F     | HR3F      |
| ------------------------ | --------- | --------- | --------- |
| **Baseline Methods**     |           |           |           |
| Harmonic-CNN*            | 0.646     | 0.571     | —         |
| SpecTNT (24 s, CTL)*     | 0.675     | 0.623     | —         |
| MusicFM-Zhang et al.*    | 0.680     | 0.636     | 0.764     |
| LinkSeg                  | 0.747     | 0.648     | 0.786     |
| TA (Zhang et al., 2025)* | 0.779     | 0.506     | 0.691     |
| **SongFormer (Ours)**    |           |           |           |
| SongFormer (HX)          | 0.787     | **0.651** | 0.795     |
| SongFormer (HX+E+H)      | **0.814** | 0.650     | **0.804** |
| SongFormer (HX+E+H+G)    | 0.812     | 0.650     | 0.800     |

- Results marked with * are taken from original papers due to unavailable implementations
- Dataset abbreviations: HX (HarmonixSet), E, H, G refer to different training datasets as stated in the paper

## Installation

### Setting up Python Environment

```bash
git clone https://github.com/ASLP-lab/SongFormer.git

# Get MuQ and MusicFM source code
git submodule update --init --recursive

conda create -n songformer python=3.10 -y
conda activate songformer
```

For users in mainland China, you may need to set up pip mirror source:

```bash
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple
```

Install dependencies:

```bash
pip install -r requirements.txt
```

We tested this on Ubuntu 22.04.1 LTS and it works normally. If you cannot install, you may need to remove version constraints in `requirements.txt`

### Download Pre-trained Models

```bash
cd src/SongFormer
# For users in mainland China, you can modify according to the py file instructions to use hf-mirror.com for downloading
python utils/fetch_pretrained.py
```

After downloading, you can verify the md5sum values in `src/SongFormer/ckpts/md5sum.txt` match the downloaded files:

```bash
md5sum ckpts/MusicFM/msd_stats.json
md5sum ckpts/MusicFM/pretrained_msd.pt
md5sum ckpts/SongFormer.safetensors
# md5sum ckpts/SongFormer.pt
```

## Inference

### 1. One-Click Inference with HuggingFace Space

Available at: [https://huggingface.co/spaces/ASLP-lab/SongFormer](https://huggingface.co/spaces/ASLP-lab/SongFormer)

### 2. Gradio App

First, change directory to the project root directory and activate the environment:

```bash
conda activate songformer
```

You can modify the server port and listening address in the last line of `app.py` according to your preference.

> If you're using an HTTP proxy, please ensure you include:
>
> ```bash
> export no_proxy="localhost, 127.0.0.1, ::1"
> export NO_PROXY="localhost, 127.0.0.1, ::1"
> ```
>
> Otherwise, Gradio may incorrectly assume the service hasn't started, causing startup to exit directly.

When first running `app.py`, it will connect to Hugging Face to download MuQ-related weights. We recommend creating an empty folder in an appropriate location and using `export HF_HOME=XXX` to point to this folder, so cache will be stored there for easy cleanup and transfer.

And for users in mainland China, you may need `export HF_ENDPOINT=https://hf-mirror.com`. For details, refer to https://hf-mirror.com/

```bash
python app.py
```

### 3. Python Code

You can refer to the file `src/SongFormer/infer/infer.py`. The corresponding execution script is located at `src/SongFormer/infer.sh`. This is a ready-to-use, single-machine, multi-process annotation script.

Below are some configurable parameters from the `src/SongFormer/infer.sh` script. You can set `CUDA_VISIBLE_DEVICES` to specify which GPUs to use:

```bash
-i              # Input SCP folder path, each line containing the absolute path to one audio file
-o              # Output directory for annotation results
--model         # Annotation model; the default is 'SongFormer', change if using a fine-tuned model
--checkpoint    # Path to the model checkpoint file
--config_pat    # Path to the configuration file
-gn             # Total number of GPUs to use — should match the number specified in CUDA_VISIBLE_DEVICES
-tn             # Number of processes to run per GPU
```

You can control which GPUs are used by setting the `CUDA_VISIBLE_DEVICES` environment variable.

> Notes
> - You may need to modify line 121 in `src/third_party/musicfm/model/musicfm_25hz.py` to:
> `S = torch.load(model_path, weights_only=False)["state_dict"]`

## Evaluation

### 1. Preparing MSA TXT Format for GT Annotations and Inference Results

The MSA TXT file format follows this structure:

```
start_time_1 label_1
start_time_2 label_2
....
end_time end
```

Each line contains two space-separated elements:

- **First element**: Timestamp in seconds (float type)
- **Second element**: Label (string type)

**Conversion Notes:**

- **SongFormer outputs** can be converted using the utility script `src/SongFormer/utils/convert_res2msa_txt.py`
- **Other annotation tools** require custom conversion to this format
- All MSA TXT files should be stored in a folder with **consistent naming** between ground truth (GT) and inference results

### 2. Supported Labels and Definitions

| ID   | Label      | Description                                                  |
| ---- | ---------- | ------------------------------------------------------------ |
| 0    | intro      | Opening section, typically appears at the beginning, rarely in middle or end |
| 1    | verse      | Main narrative section with similar melody but different lyrics across repetitions; emotionally moderate, storytelling-focused |
| 2    | chorus     | Climactic, highly repetitive section that forms the song's memorable hook; features diverse instrumentation and elevated energy |
| 3    | bridge     | Contrasting section appearing once after 2-3 choruses, providing variation before returning to verse or chorus |
| 4    | inst       | Instrumental section with minimal or no vocals, occasionally featuring speech elements |
| 5    | outro      | Closing section, typically at the end, rarely appearing in beginning or middle |
| 6    | silence    | Silent segments, usually before intro or after outro         |
| 26   | pre-chorus | Transitional section between verse and chorus, featuring additional instruments and building emotional intensity |
| -    | end        | Timestamp marker for song conclusion (not a label)           |

**Important Note**: While our model generates 8 categories, mainstream evaluation uses 7 categories. During evaluation, `pre-chorus` labels are mapped to `verse` according to our mapping rules.

### 3. Running the Evaluation

The main evaluation script is located at `src/SongFormer/evaluation/eval_infer_results.py`. You can use the shell script `src/SongFormer/eval.sh` for streamlined evaluation.

#### Parameter Configuration

| Parameter                   | Description                                                  | Default Setting |
| --------------------------- | ------------------------------------------------------------ | --------------- |
| `ann_dir`                   | Ground truth directory                                       | Required        |
| `est_dir`                   | Inference results directory                                  | Required        |
| `output_dir`                | Output directory for evaluation results                      | Required        |
| `prechorus2what`            | Mapping strategy for `pre-chorus` labels:• `verse`: Map to verse• `chorus`: Map to chorus• None: Keep original | Map to `verse`  |
| `merge_continuous_segments` | Merge consecutive segments with identical labels             | Disabled        |

## Training

Before starting, ensure you have the necessary dependencies installed and your environment properly configured.

### Step 1: Extract SSL Representations

The SSL representation extraction code is located in `src/data_pipeline`. Navigate to this directory first:

```bash
cd src/data_pipeline
```

For each song, you need to extract 4 different representations:

- **MuQ - 30s**: Short-term features with 30-second windows
- **MuQ - 420s**: Long-term features with 420-second windows
- **MusicFM - 30s**: Short-term features with 30-second windows
- **MusicFM - 420s**: Long-term features with 420-second windows

For 30-second representations, the extraction process employs a window size and hop size of 30 seconds, with features concatenated after extraction, resulting in a sequence length matching that of the 420-second version.

Run the following scripts after configuring them for your environment:

```bash
# MuQ representations
bash obtain_SSL_representation/MuQ/get_embeddings_30s_wrap420s.sh
bash obtain_SSL_representation/MuQ/get_embeddings.sh

# MusicFM representations  
bash obtain_SSL_representation/MusicFM/get_embeddings_mp_30s_wrap420s.sh
bash obtain_SSL_representation/MusicFM/get_embeddings_mp.sh
```

### Step 2: Configure Training Parameters

Edit `src/SongFormer/configs/SongFormer.yaml` to set:

- `train_dataset`: Training dataset configuration
- `eval_dataset`: Evaluation dataset configuration
- `args`: Model settings and experiment name

------

For the `dataset_abstracts` class, configure these parameters:

| Parameter             | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `internal_tmp_id`     | Unique identifier for the dataset instance                   |
| `dataset_type`        | Dataset ID from `src/SongFormer/dataset/label2id.py` (see `DATASET_LABEL_TO_DATASET_ID`) |
| `input_embedding_dir` | Space-separated paths to four SSL representation folders     |
| `label_path`          | Path to JSONL file with labels (see [example format](https://huggingface.co/datasets/ASLP-lab/SongFormDB/blob/main/data/Gem/SongFormDB-Gem.jsonl)) |
| `split_ids_path`      | Text file with one ID per line specifying data to use (IDs not in this file will be ignored) |
| `multiplier`          | Data balancing factor - repeats small datasets to match larger ones |

-----

Update `src/SongFormer/train/accelerate_config/single_gpu.yaml` with your desired accelerate settings, and configure `src/SongFormer/train.sh` accordingly:

- Your Weights & Biases (wandb) API key
- Other training-specific settings

### Step 3: Run Training

Navigate to the SongFormer directory and execute the training script:

```bash
cd src/SongFormer
bash train.sh
```

- The relevant training dashboard will be displayed on `wandb`
- Checkpoints will be located in `src/SongFormer/output`

## Citation

If our work and codebase is useful for you, please cite as:

````bibtex
@misc{hao2026songformerscalingmusicstructure,
      title={SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision}, 
      author={Chunbo Hao and Ruibin Yuan and Jixun Yao and Qixin Deng and Xinyi Bai and Yanbo Wang and Wei Xue and Lei Xie},
      year={2026},
      eprint={2510.02797},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2510.02797}, 
}
````

## License

Our code is released under CC-BY-4.0 License.

## Contact Us

We welcome your feedback and contributions! You can reach us through:

- **Report Issues:** Found a bug or have a suggestion? Please open an issue directly in this GitHub repository. This is the best way to track and resolve problems.
- **Join Our Community:** For discussions and real-time support, join our Discord server: https://discord.gg/rwcqh7Em

We look forward to hearing from you!

<p align="center">
    <a href="http://www.nwpu-aslp.org/">
        <img src="figs/aslp.png" width="400"/>
    </a>
</p>
