# Rethinking Diffusion for Text-Driven Human Motion Generation: Redundant Representations, Evaluation, and Masked Autoregression (CVPR 2025)
![](./MARDM.png)

<p align="center">
  <a href='https://arxiv.org/abs/2411.16575'>
    <img src='https://img.shields.io/badge/Arxiv-2411.16575-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/abs/2411.16575.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href='https://neu-vi.github.io/MARDM/'>
  <img src='https://img.shields.io/badge/Project-Page-orange?style=flat&logo=Google%20chrome&logoColor=orange'></a>
  <a href='https://github.com/neu-vi/MARDM'>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=neu-vi.MARDM&left_color=gray&right_color=blue">
  </a>
</p>

<p align="center">
<strong>Rethinking Diffusion for Text-Driven Human Motion Generation</strong></h1>
   <p align="center">
    <a href='https://cr8br0ze.github.io' target='_blank'>Zichong Meng</a>&emsp;
    <a href='https://ymingxie.github.io/' target='_blank'>Yiming Xie</a>&emsp;
    <a href='https://xiaogangpeng.github.io/' target='_blank'>Xiaogang Peng</a>&emsp;
    <a href='https://show-han.github.io/' target='_blank'>Zeyu Han</a>&emsp;
    <a href='https://jianghz.me/' target='_blank'>Huaizu Jiang</a>&emsp;
    <br>
    Northeastern University 
    <br>
    CVPR 2025
  </p>
</p>

### [NEW] [ACMDM](https://neu-vi.github.io/ACMDM/): Absolute Coordinates Are All We Need for Text-Driven Motion Generation

### Official Simple & Minimalist PyTorch Implementation

## 📜 TODO List
- [x] Release the clean codes for implementation.
- [x] Release the evaluation codes and the pretrained models.
- [x] Release the simple and minimalist version of codes for implementation.
- [ ] Release updated version of AE weights and scripts

## 📢 News
- Will be releasing the updated version of AE weights with more support and scripts soon after cleaning the code.

##  ⚙️ Getting Started
<details>
  
### 1. Conda Environment
```bash
conda env create -f environment.yml
conda activate MARDM
```
We test our code on Python 3.10.13, PyTorch 2.2.0, and CUDA 12.1

### 2. Models and Dependencies

Due to the deactivation of the previous Google account, we have migrated all model files to [Hugging Face](https://huggingface.co/collections/cr8br0ze/mardm).

#### Download Evaluation Models
```bash
rm -rf checkpoints
mkdir checkpoints
cd checkpoints
mkdir t2m
mkdir kit

cd t2m
echo -e "Downloading evaluation models for HumanML3D dataset"
wget -c \
  "https://huggingface.co/cr8br0ze/evaluators_humanml3d/resolve/main/evaluators_humanml3d.zip" \
  -O evaluators_humanml3d.zip

echo -e "Unzipping humanml3d evaluators"
unzip evaluators_humanml3d.zip

echo -e "Cleaning humanml3d evaluators zip"
rm evaluators_humanml3d.zip

cd ../kit/
echo -e "Downloading pretrained models for KIT-ML dataset"
wget -c \
  "https://huggingface.co/cr8br0ze/evaluators_kit/resolve/main/evaluators_kit.zip" \
  -O evaluators_kit.zip

echo -e "Unzipping kit evaluators"
unzip evaluators_kit.zip

echo -e "Cleaning kit evaluators zip"
rm evaluators_kit.zip

cd ../../
```

#### Download GloVe
```bash
rm -rf glove
echo -e "Downloading glove (in use only by the evaluators)"
gdown --fuzzy https://drive.google.com/file/d/1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n/view?usp=sharing

unzip glove.zip
echo -e "Cleaning GloVe zip\n"
rm glove.zip

echo -e "Downloading done!"
```

#### Download Pre-trained Models
```bash
cd checkpoints/t2m
echo -e "Downloading pretrained models for HumanML3D dataset"
wget -c \
  "https://huggingface.co/cr8br0ze/MARDM_SiT_XL/resolve/main/MARDM_SiT_XL.zip" \
  -O MARDM_SiT_XL.zip
wget -c \
  "https://huggingface.co/cr8br0ze/MARDM_DDPM_XL/resolve/main/MARDM_DDPM_XL.zip" \
  -O MARDM_DDPM_XL.zip
wget -c \
  "https://huggingface.co/cr8br0ze/length_estimator/resolve/main/length_estimator.zip" \
  -O length_estimator.zip
wget -c \
  "https://huggingface.co/cr8br0ze/AE_humanml3d/resolve/main/AE_humanml3d.zip" \
  -O AE_humanml3d.zip
echo -e "Unzipping"
unzip MARDM_SiT_XL.zip
unzip MARDM_DDPM_XL.zip
unzip length_estimator.zip
unzip AE_humanml3d.zip
echo -e "Cleaning zips"
rm MARDM_SiT_XL.zip
rm MARDM_DDPM_XL.zip
rm length_estimator.zip
rm AE_humanml3d.zip

cd ../../
```

### 3. Obtain Data
**You do not need to get data** if you only want to generate motions based on textual instructions.

If you want to reproduce and evaluate our method, you can obtain both 
**HumanML3D** and **KIT** following instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git). By default, the data path is set to `./datasets`.

For dataset Mean and Std, you are welcome to use the eval_mean,npy and eval_std,npy in the utils,
or you can calculate based on your obtained dataset using:
```
python utils/cal_mean_std.py
```
</details>

## 💻  Demo
<details>

### (a) Generate with single textual instruction
```bash
python sample.py --name MARDM_SiT_XL --text_prompt "A person is running on a treadmill."
```
### (b) Generate from a prompt file
in a txt file, in each line, your input should be `<text description>#<motion length>`,
you can push NA as motion length to let model determine the motion length
(if there is **one** NA in file, all the others will be **NA** as well).

```bash
python sample.py --name MARDM_SiT_XL --text_path ./text_prompt.txt
```
</details>

## 🎆 Train Your Own MARDM models
<details>

### HumanML3D
#### AE
```bash
python train_AE.py --name AE --dataset_name t2m --batch_size 256 --epoch 50 --lr_decay 0.05
```
#### MARDM
```bash
# MARDM SiT-based (best results)
python train_MARDM.py --name MARDM_SiT_XL --model "MARDM-SiT-XL" --dataset_name t2m --batch_size 64 --ae_name AE
# MARDM DDPM-based
python train_MARDM.py --name MARDM_DDPM_XL --model "MARDM-DDPM-XL" --dataset_name t2m --batch_size 64 --ae_name AE
```

### KIT-ML
#### AE
```bash
python train_AE.py --name AE --dataset_name kit --batch_size 512 --epoch 50 --lr_decay 0.1
```
#### MARDM
```bash
# MARDM SiT-based (best results)
python train_MARDM.py --name MARDM_SiT_XL --model "MARDM-SiT-XL" --dataset_name kit --batch_size 16 --ae_name AE --milestones 20000
# MARDM DDPM-based
python train_MARDM.py --name MARDM_DDPM_XL --model "MARDM-DDPM-XL" --dataset_name kit --batch_size 16 --ae_name AE --milestones 20000
```
</details>

## 📖 Evaluate MARDM models
<details>

### HumanML3D
#### AE
```bash
python evaluation_AE.py --name AE --dataset_name t2m
```
#### MARDM
```bash
# MARDM SiT-based (best results)
python evaluation_MARDM.py --name MARDM_SiT_XL --model "MARDM-SiT-XL" --dataset_name t2m --cfg 4.5
# MARDM DDPM-based
python evaluation_MARDM.py --name MARDM_DDPM_XL --model "MARDM-DDPM-XL" --dataset_name t2m --cfg 4.5
```
### KIT-ML
#### AE
```bash
python evaluation_AE.py --name AE --dataset_name kit
```
#### MARDM
```bash
# MARDM SiT-based (best results)
python evaluation_MARDM.py --name MARDM_SiT_XL --model "MARDM-SiT-XL" --dataset_name kit --cfg 2.5
# MARDM DDPM-based
python evaluation_MARDM.py --name MARDM_DDPM_XL --model "MARDM-DDPM-XL" --dataset_name kit --cfg 2.5
```
</details>

## 🎏 Temporal Editing
<details>

```bash
python edit.py --name MARDM_SiT_XL -msec 0.3,0.6 --text_prompt "A man dancing around." --source_motion 000612.npy
```
</details>

## 🍀 Acknowledgments
This code is standing on the shoulders of giants, we would like to thank the following contributors that our code is based on:.

Our original raw implementation is heavily based on [T2M](https://github.com/EricGuo5513/text-to-motion),
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MMM](https://github.com/exitudio/MMM) 
and [MoMask](https://github.com/EricGuo5513/momask-codes).
The Diffusion part is primarily based on [DDPM](https://github.com/hojonathanho/diffusion),
[DiT](https://github.com/facebookresearch/DiT), [SiT](https://github.com/willisma/SiT),
[MAR](https://github.com/LTH14/mar/), [HOI-Diff](https://github.com/neu-vi/HOI-Diff),
[InterGen](https://github.com/tr3e/InterGen), [MDM](https://github.com/GuyTevet/motion-diffusion-model),
[MLD](https://github.com/ChenFengYe/motion-latent-diffusion).

For open sourced version, we decide to restructure (and some rewrite) for a simple and minimalist version of PyTorch code implementation
that get rids of PyTorch Lighting implicit hooks, outer-space variable utilization and implicit argparse calls.
We hope our minimalist version implementation can lead to better code comprehension and contribution to the motion generation community. Thank you.

## 🤝 Citation
If you find this repository useful for your work, please consider citing it as follows:
```bibtex
@article{meng2024rethinking,
      title={Rethinking Diffusion for Text-Driven Human Motion Generation},
      author={Meng, Zichong and Xie, Yiming and Peng, Xiaogang and Han, Zeyu and Jiang, Huaizu},
      journal={arXiv preprint arXiv:2411.16575},
      year={2024}
    }
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neu-vi/MARDM&type=Date)](https://star-history.com/#neu-vi/MARDM&Date)
