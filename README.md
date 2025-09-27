<p align="center">
  <img src="figs/logo.png" width="50%" />
</p>


# SONGFORMER: SCALING MUSIC STRUCTURE ANALYSIS WITH HETEROGENEOUS SUPERVISION

![Python](https://img.shields.io/badge/Python-3.10-brightgreen)  
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightblue)  
[![arXiv](https://img.shields.io/badge/arXiv-com.svg?logo=arXiv)]()  
[![GitHub](https://img.shields.io/badge/GitHub-SongFormer-black)](https://github.com/ASLP-lab/SongFormer)  
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-space-yellow)](https://huggingface.co/spaces/ASLP-lab/SongFormer)  
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-model-blue)](https://huggingface.co/ASLP-lab/SongFormer)  
[![Dataset SongFormDB](https://img.shields.io/badge/HF%20Dataset-SongFormDB-green)](https://huggingface.co/datasets/ASLP-lab/SongFormDB)  
[![Dataset SongFormBench](https://img.shields.io/badge/HF%20Dataset-SongFormBench-orange)](https://huggingface.co/datasets/ASLP-lab/SongFormBench)
[![Discord](https://img.shields.io/badge/Discord-join%20us-purple?logo=discord&logoColor=white)](https://discord.gg/rwcqh7Em)
[![lab](https://img.shields.io/badge/🏫-ASLP-grey?labelColor=lightgrey)](http://www.npu-aslp.org/)

Chunbo Hao<sup>&ast;</sup>, Ruibin Yuan<sup>&ast;</sup>, Jixun Yao, Qixin Deng, Xinyi Bai, Wei Xue, Lei Xie<sup>&dagger;</sup>


----


SongFormer is a music structure analysis framework that leverages multi-resolution self-supervised representations and heterogeneous supervision, accompanied by the large-scale multilingual dataset SongFormDB and the high-quality benchmark SongFormBench to foster fair and reproducible research.

![](figs/songformer.png)

## News and Updates

## 📋 To-Do List

- [ ] Complete and push inference code to GitHub
- [ ] Upload model checkpoint(s) to Hugging Face Hub
- [ ] Upload the paper to arXiv
- [ ] Fix readme
- [ ] Deploy an out-of-the-box inference version on Hugging Face (via Inference API or Spaces)
- [ ] Publish the package to PyPI for easy installation via `pip`
- [ ] Open-source evaluation code
- [ ] Open-source training code

## Installation

comming soon

## Inference

### 1. Gradio App

comming soon

### 2. CLI Inference

comming soon

### 3. Python API

comming soon

### 4. Pitfall

- You may need to modify line 121 in `src/third_party/musicfm/model/musicfm_25hz.py` to:
`S = torch.load(model_path, weights_only=False)["state_dict"]`

## Training

## Citation

If our work and codebase is useful for you, please cite as:

````
comming soon
````
## License

Our code is released under CC-BY-4.0 License.

## Contact Us

If you are interested in leaving a message to our research team, feel free to email `nzqiann@gmail.com`.
<p align="center">
    <a href="http://www.nwpu-aslp.org/">
        <img src="figs/aslp.png" width="400"/>
    </a>
</p>


