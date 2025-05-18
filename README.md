# TDVE-Assessor: Benchmarking and Evaluating the Quality of Text-Driven Video Editing with LMMs

[![Paper NeurIPS 2025](https://img.shields.io/badge/Paper-NeurIPS%202025-B31B1B.svg)](https://link_to_your_paper.com) [![Dataset TDVE-DB](https://img.shields.io/badge/Dataset-TDVE--DB-blue.svg)](https://huggingface.co/datasets/Moyao001/TDVE-DB/tree/main)
[![License](https://img.shields.io/badge/License-CC--BY--SA--4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/) Official PyTorch implementation for **TDVE-Assessor**, a novel Large Multimodal Model (LMM) based approach for evaluating the quality of text-driven video editing, as presented in our NeurIPS 2025 paper: "[TDVE-Assessor: Benchmarking and Evaluating the Quality of Text-Driven Video Editing with LMMs](https://link_to_your_paper.com)".

TDVE-Assessor integrates spatial and temporal video features into an LMM (Qwen2.5-VL-7B) for rich contextual understanding to provide comprehensive quality assessment across three crucial dimensions:
1.  Edited Video Quality
2.  Editing Alignment (Text-Video Consistency)
3.  Structural Consistency

<p align="center">
  <img src="image/MODEL.png" alt="TDVE-Assessor Model Overview" width="700"/>
</p>
<p>
(a) Video features are extracted by a frozen spatiotemporal encoder and aligned to a Large Language Model (Qwen2.5-VL-7B) using trainable projection modules. The LLM, fine-tuned with LoRA, then generates textual feedback for multiple evaluation dimensions via a trainable text decoder, while its last hidden states (quality representations) are input to a quality regression module. The model also supports pairwise comparison fine-tuning. (b) The quality regression module, a trainable MLP, converts these LLM-derived quality representations into numerical scores for dimensions like video quality, editing alignment, and structural consistency.
</p>

We also introduce **TDVE-DB**, the largest and most comprehensive benchmark dataset for text-driven video editing quality assessment, featuring 3,857 edited videos from 12 models across 8 editing categories, annotated with 173,565 human subjective ratings.
<p align="center">
  <img src="image/dataset.png" alt="TDVE-DB Dataset Overview" width="700"/>
</p>
<p>
(a) Acquisition of the source video. (b) Generation of prompt words. (c) Obtaining 170K subjective scores through subjective experiments. (d) The number of videos for different models and different editing categories. (e) Three-dimensional scatter plot of subjective scores.
</p>

## 📝 Table of Contents

- [TDVE-Assessor: Benchmarking and Evaluating the Quality of Text-Driven Video Editing with LMMs](#tdve-assessor-benchmarking-and-evaluating-the-quality-of-text-driven-video-editing-with-lmms)
  - [📝 Table of Contents](#-table-of-contents)
  - [✨ Highlights](#-highlights)
  - [📊 Key Data from the Paper](#-key-data-from-the-paper)
    - [Overview of Text-Driven Video Editing Models in TDVE-DB](#overview-of-text-driven-video-editing-models-in-tdve-db)
    - [Comparison of Video Editing Datasets](#comparison-of-video-editing-datasets)
    - [Performance Benchmark on TDVE-DB (Full Table)](#performance-benchmark-on-tdve-db-full-table)
    - [Cross-Dataset Evaluation Performance](#cross-dataset-evaluation-performance)
    - [Ablation Study: Comparison of Training Strategies](#ablation-study-comparison-of-training-strategies)
    - [Inter-Rater Reliability (ICC) of Subjective Scores in TDVE-DB](#inter-rater-reliability-icc-of-subjective-scores-in-tdve-db)
    - [Average Score Heatmaps by Model and Category (Summary)](#average-score-heatmaps-by-model-and-category-summary)
    - [Project URLs for Benchmarked Models](#project-urls-for-benchmarked-models)
  - [🚀 Getting Started](#-getting-started)
    - [Installation](#installation)
  - [📊 Dataset (TDVE-DB)](#-dataset-tdve-db)
  - [🤖 Model (TDVE-Assessor)](#-model-tdve-assessor)
    - [Model Architecture](#model-architecture)
    - [Pretrained Models](#pretrained-models)
  - [⚙️ Usage](#️-usage)
    - [Data Preparation](#data-preparation)
    - [Train](#train)
    - [Evaluation](#evaluation)

## ✨ Highlights

* **TDVE-Assessor**: A novel LMM-based model (Qwen2.5-VL-7B backbone) achieving SOTA performance for text-driven video editing quality assessment.
* **TDVE-DB**: The largest benchmark dataset for this task, with extensive multi-dimensional human annotations.
* Comprehensive evaluation of 12 state-of-the-art video editing models.
* Code and dataset publicly available to foster future research.

## 📊 Key Data from the Paper

This section summarizes crucial tables and data presented in the research paper.

### Overview of Text-Driven Video Editing Models in TDVE-DB

The following table details the text-driven video editing models used for constructing the TDVE-DB dataset.
*SD means Stable Diffusion. 'Follow Source' means the output video maintains the same resolution as the source video.*

| Model             | Year  | Length | Base Model | Resolution    | FPS | Zero-shot | Open Source |
| :---------------- | :---- | :----- | :--------- | :------------ | :-- | :-------- | :---------- |
| Tune-A-Video   | 22.12 | 3s     | SD 1-4     | $512 \times 512$ | 8   | X         | ✓           |
| Tokenflow    | 23.07 | 1s     | SD 2-1     | $512 \times 512$ | 30  | ✓         | ✓           |
| Text2Video-Zero | 23.03 | 1-5s   | SD 1-5     | Follow Source | 24  | ✓         | ✓           |
| CCEdit       | 23.09 | 2s     | SD 1-5     | $768 \times 512$ | 6   |     ✓      | ✓           |
| ControlVideo   | 23.05 | 1s     | SD 1-5     | $512 \times 512$ | 8   | ✓         | ✓           |
| FateZero    | 23.03 | 1-4s   | SD 1-4     | $512 \times 512$ | 10  | ✓         | ✓           |
| FLATTEN        | 23.12 | 1-2s   | SD 2-1     | $512 \times 512$ | 15  | ✓         | ✓           |
| FRESCO         | 24.06 | 1-5s   | SD 1-5     | Follow Source | 24  | ✓         | ✓           |
| Pix2Video     | 23.03 | 2-4s   | SD 2       | $512 \times 512$ | 30  | ✓         | ✓           |
| RAVE        | 23.12 | 1-3s   | SD 1-5     | Follow Source | 30  | ✓         | ✓           |
| Slicedit     | 24.05 | 1-4s   | SD 1-5     | Follow Source | 25  |     ✓      | ✓           |
| vid2vid-zero  | 23.03 | 2s     | SD 2-1     | $512 \times 512$ | 8   | ✓         | ✓           |

### Comparison of Video Editing Datasets

TDVE-DB is compared with other existing video editing datasets based on key metrics and features.

| Dataset                | Samples | Edited | MOS      | Edit Dimension | Evaluate Dimension | Count of video editing models | FPS   | Open Source |
| :--------------------- | :------ | :----- | :------- | :------------- | :----------------- | :---------------------------- | :---- | :---------- |
| Señorita-2M       | 2M      | X      | X        | X              | -                  | 4                             | 28    |      **✓**       |
| LOVEU-TGVE-2023   | 78      | ✓      | X        | X              | -                  | X                             | 8     |      **✓**       |
| BalanceCC              | 412     | ✓      | X        | 4              | -                  | 1                             | 8     |       **✓**      |
| VE-Bench DB       | 1170    | ✓      | 28,000   | 3              | 1                  | 8                             | 8     |      **✓**       |
| **TDVE-DB (Ours)** | **3857**| **✓** | **173,565**| **8** | **3** | **12** | **6-30**| **✓** |

### Performance Benchmark on TDVE-DB (Full Table)

好的，我明白了您的要求。我会重新处理Table 3，去除模型名称后的引用数字，并取消所有加粗和颜色标注，然后将其转换为标准的Markdown格式。

请稍候，我正在为您生成新的表格。

这是更新后的Table 3，已按照您的要求进行了修改：

### Performance Benchmark on TDVE-DB (Full Table)

This table showcases the performance of TDVE-Assessor against various vision-language pre-training models, open-source LLM-based models, conventional handcrafted metrics, and deep learning-based VQA models on the TDVE-DB dataset. Performance is measured by Spearman Rank Correlation Coefficient (SRCC), Pearson Linear Correlation Coefficient (PLCC), and Kendall Rank Correlation Coefficient (KRCC).

| Method                 | Video Quality \<br\> SRCC    PLCC    KRCC | Editing Alignment \<br\> SRCC    PLCC    KRCC | Structural Consistency \<br\> SRCC    PLCC    KRCC | Overall Average \<br\> SRCC    PLCC    KRCC |
| :--------------------- | :-------------------: | :--------------------: | :-----------------------: | :--------------------: |
| ImageReward            | 0.0490  0.0369  0.0330  | 0.0858  0.0736  0.0578   | 0.0179  0.0202  0.0114      | 0.0509  0.0436  0.0354   |
| BLIPScore              | 0.0910  0.0915  0.0610  | 0.0408  0.0397  0.0271   | 0.0743  0.0742  0.0497      | 0.0687  0.0685  0.0459   |
| CLIPScore              | 0.0222  0.0173  0.0114  | 0.1970  0.2086  0.1343   | 0.0210  0.0185  0.0264      | 0.0801  0.0815  0.0574   |
| PickScore              | 0.3177  0.3084  0.2153  | 0.0037  0.0141  0.0026   | 0.1147  0.1183  0.0802      | 0.1454  0.1469  0.0994   |
| VQAScore               | 0.0339  0.0586  0.0396  | 0.3761  0.3627  0.2613   | 0.0830  0.0685  0.1015      | 0.1643  0.1633  0.1341   |
| AestheticScore         | 0.2224  0.2132  0.1515  | 0.2499  0.2341  0.1713   | 0.2195  0.2309  0.1576      | 0.2306  0.2261  0.1601   |
| LLava-NEXT             | 0.1038  0.0975  0.0835  | 0.1109  0.1111  0.0886   | 0.1274  0.1446  0.1185      | 0.1140  0.1177  0.0969   |
| InternVideo2.5         | 0.3650  0.3290  0.2922  | 0.6004  0.5806  0.4802   | 0.3487  0.3608  0.2872      | 0.4380  0.4235  0.3532   |
| VideoLLAMA3            | 0.2329  0.1636  0.1016  | 0.5082  0.5118  0.3862   | 0.3267  0.2927  0.2177      | 0.3559  0.3227  0.2352   |
| InternVL               | 0.3613  0.3706  0.2518  | 0.5756  0.5916  0.4151   | 0.2427  0.3542  0.1400      | 0.3932  0.4388  0.2690   |
| mPLUG-OW13             | 0.1589  0.0977  0.2273  | 0.3470  0.3237  0.2381   | 0.1779  0.2133  0.1575      | 0.2279  0.2116  0.2076   |
| HOST                  | 0.0556  0.0599  0.0374  | 0.0440  0.0578  0.0293   | 0.0678  0.0614  0.0412      | 0.0558  0.0597  0.0360   |
| NIQE                  | 0.0856  0.0846  0.0568  | 0.0457  0.0256  0.0418   | 0.0978  0.1134  0.0698      | 0.0764  0.0745  0.0561   |
| BRISQUE                | 0.0774  0.0661  0.0513  | 0.1253  0.1005  0.0839   | 0.0316  0.0352  0.0235      | 0.0781  0.0673  0.0529   |
| BMPRI                 | 0.2665  0.0204  0.0136  | 0.1568  0.2731  0.1853   | 0.1382  0.1441  0.1006      | 0.1872  0.1459  0.0998   |
| QAC                   | 0.1385  0.1739  0.1252  | 0.0901  0.2593  0.1924   | 0.2847  0.0973  0.0984      | 0.1711  0.1768  0.1387   |
| BPRI                   | 0.3213  0.0223  0.0150  | 0.2077  0.1884  0.1362   | 0.1796  0.1838  0.1223      | 0.2362  0.1315  0.0912   |
| VSFA                   | 0.7670  0.7732  0.5784  | 0.3775  0.3999  0.2603   | 0.6392  0.6425  0.4674      | 0.5946  0.6052  0.4354   |
| BVQA                   | 0.7522  0.7755  0.5635  | 0.2984  0.3382  0.2059   | 0.7863  0.7877  0.5884      | 0.6123  0.6338  0.4526   |
| SimpleVQA              | 0.7807  0.7810  0.5717  | 0.3215  0.3839  0.2511   | 0.6300  0.6352  0.4862      | 0.5774  0.6000  0.4363   |
| FAST-VQA               | 0.8096  0.8155  0.6203  | 0.4279  0.4684  0.3017   | 0.7925  0.7954  0.5861      | 0.6767  0.6931  0.5027   |
| DOVER                  | 0.7810  0.7846  0.5775  | 0.3670  0.3018  0.3229   | 0.8001  0.8042  0.6021      | 0.6494  0.6302  0.5008   |
| TDVE-Assessor (Ours)   | 0.8688  0.8688  0.6919  | 0.8254  0.8330  0.6460   | 0.8354  0.8523  0.6564      | 0.8432  0.8514  0.6648   |


### Cross-Dataset Evaluation Performance

TDVE-Assessor's generalization capabilities tested on other public benchmarks.

| Benchmark Dataset | Model                    | SRCC            | PLCC            | KRCC            |
| :---------------- | :----------------------- | :-------------- | :-------------- | :-------------- |
| **VE-Bench DB**| VE-Bench QA         | 0.7330          | 0.7415          | 0.5414          |
|                   | **TDVE-Assessor (Ours)** | **0.7527**| **0.7654**| **0.5645**|
| **T2VQA-DB** | T2VQA               | 0.8179          | 0.8227          | 0.6370          |
|                   | **TDVE-Assessor (Ours)** | **0.8222**| **0.8335**| **0.6399**|
| **AIGVQA-DB** | AIGV-Assessor     | 0.9162          | 0.9190          | 0.7576          |
|                   | **TDVE-Assessor (Ours)** | **0.9397**| **0.9344**| **0.7835**|

*(Other compared models in the paper for VE-Bench DB include CLIP-F, Sadit, PickScore, FAST-VOA, DOVER. For T2VQA-DB, models include UMTScore, SimpleVQA, FAST-VQA, BVQA, DOVER, Q-Align. For AIGVQA-DB, models include VSFA, SimpleVQA, FAST-VQA, BVQA, DOVER. TDVE-Assessor shows SOTA or competitive performance.)*

### Ablation Study: Comparison of Training Strategies

This table shows the impact of different training strategies on TDVE-Assessor's performance. `Exp.5` was chosen as the default configuration.

| Exp.  | LORA (vision) | LORA (LLM) | Quality Regression | Video Quality SRCC | Editing Alignment SRCC | Structural Consistency SRCC |
| :---- | :------------ | :--------- | :----------------- | :----------------- | :----------------------- | :-------------------------- |
| Exp.1 | ✓             |            |                    | 0.1445             | 0.3556                   | 0.3874                      |
| Exp.2 |               | ✓          |                    | 0.7633             | 0.7968                   | 0.7901                      |
| Exp.3 |               |            | ✓                  | 0.6201             | 0.6177                   | 0.6255                      |
| Exp.4 | ✓             | ✓          |                    | 0.7641             | 0.7966                   | 0.7785                      |
| **Exp.5**|           | **✓** | **✓** | **0.8688** | **0.8254** | **0.8354** |
| Exp.6 | ✓             | ✓          | ✓                  | 0.8679             | 0.8241                   | 0.8321                      |

*The PLCC and KRCC values for each experiment are available in Table 5 of the paper.*

### Inter-Rater Reliability (ICC) of Subjective Scores in TDVE-DB

The reliability and consistency of the subjective scores in TDVE-DB were analyzed using Intraclass Correlation Coefficient (ICC).

| Evaluation Dimension   | ICC2  | 95% CI (ICC2) | ICC2k | 95% CI (ICC2k) | ICC2 Level* | MOS Reliability (ICC2k) |
| :--------------------- | :---- | :------------ | :---- | :------------- | :---------- | :---------------------- |
| Video Quality          | 0.701 | (0.65, 0.75)  | 0.955 | (0.94, 0.97)   | Good        | Excellent               |
| Editing Alignment      | 0.753 | (0.71, 0.80)  | 0.920 | (0.89, 0.94)   | Excellent   | Excellent               |
| Structural Consistency | 0.685 | (0.63, 0.74)  | 0.945 | (0.92, 0.96)   | Good        | Excellent               |

*\*ICC2 Level interpretation: e.g., Good, Excellent.*

### Average Score Heatmaps by Model and Category (Summary)

Figures 8, 9, 10, and 11 in the paper present heatmaps showing the average scores of the 12 video editing models across the 8 editing categories for each of the three dimensions (Video Quality, Editing Alignment, Structural Consistency) and the overall average.

**Top Performing Models (General Trend from Heatmaps):**
* **Slicedit**, **FRESCO**, and **FateZero** generally show strong performance in **Video Quality**.
* **RAVE**, **Tokenflow**, and **Pix2video** tend to perform well in **Editing Alignment**.
* **Slicedit** and **FRESCO** also excel in **Structural Consistency**.

**Editing Category Observations:**
* **Color** editing is often a high-performing category across dimensions.
* Single-entity edits (e.g., "color") generally score higher than multi-entity edits (e.g., "multi-object").

*(For detailed heatmaps and specific scores, please refer to Figures 8-11 in the supplementary material of the paper.)*

### Project URLs for Benchmarked Models

**Text-Driven Video Editing Models:**

| Method             | URL                                               |
| :----------------- | :------------------------------------------------ |
| Tune-A-Video    | `https://github.com/showlab/Tune-A-Video`         |
| Tokenflow       | `https://github.com/omerbt/TokenFlow`             |
| Text2Video-Zero | `https://github.com/Picsart-AI-Research/Text2Video-Zero` |
| CCEdit          | `https://github.com/RuoyuFeng/CCEdit`             |
| ControlVideo    | `https://github.com/YBYBZhang/ControlVideo`       |
| FateZero        | `https://github.com/ChenyangQiQi/FateZero`        |
| FLATTEN         | `https://github.com/yrcong/FLATTEN`               |
| FRESCO         | `https://github.com/williamyang1991/FRESCO`        |
| Pix2Video       | `https://github.com/duyguceylan/pix2video`        |
| RAVE         | `https://github.com/rehglab/RAVE`                 |
| Slicedit       | `https://github.com/fallenshock/Slicedit`         |
| vid2vid-zero   | `https://github.com/baaivision/vid2vid-zero`      |

**Video Quality Assessment Methods / Metrics:**

| Method             | URL                                                     |
| :----------------- | :------------------------------------------------------ |
| ImageReward    | `https://github.com/THUDM/ImageReward`                  |
| BLIPScore      | `https://github.com/salesforce/BLIP`                    |
| CLIPScore      | `https://github.com/jmhessel/clipscore`                 |
| PickScore     | `https://github.com/yuvalkirstain/PickScore`            |
| VQAScore       | `https://github.com/linzhiqiu/t2v_metrics`              |
| AestheticScore | `https://github.com/sorekdj60/AestheticScore`           |
| LLAVA-NEXT     | `https://github.com/LLAVA-VL/LLAVA-NeXT`                |
| InternVideo2.5 | `https://github.com/OpenGVLab/InternVideo`              |
| VideoLLAMA3    | `https://github.com/DAMO-NLP-SG/VideoLLaMA3`            |
| InternVL       | `https://github.com/OpenGVLab/InternVL`                 |
| mPLUG-OWL3     | `https://github.com/X-PLUG/mPLUG-Owl`                   |
| VSFA           | `https://github.com/lidq92/VSFA`                        |
| BVQA           | `https://github.com/vztu/BVQA_Benchmark`                |
| SimpleVQA      | `https://github.com/sunwei925/SimpleVQA`                |
| FAST-VQA       | `https://github.com/timothyhtimothy/FAST-VQA-and-FasterVQA` |
| DOVER          | `https://github.com/VQAssessment/DOVER`                 |

## 🚀 Getting Started

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JuntongWang/TDVE-Assessor.git](https://github.com/JuntongWang/TDVE-Assessor.git)
    cd TDVE-Assessor
    ```

2.  **Create a virtual environment (recommended):**
    Using Conda:
    ```bash
    conda create -n tdve_assessor python=3.10 -y
    conda activate tdve_assessor
    ```

3.  **Install dependencies:**
    We provide a `requirements.txt` file with the necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

    *(For specific PyTorch installation with CUDA, refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for commands tailored to your system.)*

## 📊 Dataset (TDVE-DB)

Our TDVE-DB dataset is central to this work.
* **Content**: 3,857 edited videos, 12 editing models, 8 categories, 340 prompts, 173,565 human ratings.
* **Dimensions**: Edited Video Quality, Editing Alignment, Structural Consistency.
* **Access**: The dataset is publicly available on Hugging Face:
    [![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-TDVE--DB-blue)](https://huggingface.co/datasets/Moyao001/TDVE-DB/tree/main)

Please refer to the Hugging Face page and our paper for detailed information on data structure, collection, and statistics.

## 🤖 Model (TDVE-Assessor)

### Model Architecture

TDVE-Assessor leverages the Qwen2.5-VL-7B Instruct model as its backbone. It processes video frames using the embedded ViT and integrates these visual features with textual prompts (and source video information for structural consistency) within the LMM. A lightweight MLP regression head is then used to predict continuous quality scores for the three dimensions.

For more details, please see Section 4 of our paper.

### Pretrained Models

We plan to release the pretrained weights for TDVE-Assessor. Check back soon for download links and instructions.
* `tdve_assessor_video_quality.pth`
* `tdve_assessor_editing_alignment.pth`
* `tdve_assessor_structural_consistency.pth`

## ⚙️ Usage

### Data Preparation

1.  **Download TDVE-DB**: Follow the instructions [here](#-dataset-tdve-db) to download and prepare the TDVE-DB dataset.
2.  **Organize Data**: "You can find the corresponding training and testing JSON files in the `stage1_dataset` and `stage2_dataset` folders. You will need to modify the root directory paths within these files to match your own terminal's setup."
    ```
    TDVE-DB/
    ├── color/
    │   ├── edit_video/
    │   │   ├── fresco ├──1000.mp4
    │   │   ├── ccedit └──1001.mp4
    │   │   └── ...
    │   └── ...
    │
    └── ...
    ```
3.  If you want to use your own dataset, it's also quite simple. You just need to follow the format of our provided JSON files and add your prompts and video paths accordingly.

### Train

**Stage1:**
```bash
bash train_LLM.py
```
Then, merge the model weights well-trained in the first stage with the initial weights (the initial weights can be downloaded from Qwen/Qwen2.5-VL-7B-Instruct on Hugging Face). The merge requires calling:
```
bash merge_lora.sh
```
**Stage2:**
```
bash train_MLP.sh
```
### Evaluate
You can evaluate videos using the pretrained TDVE-Assessor model. The pretrained model will come soon.
Just use
```
python ./evaluate/eval_auth.py
```
What you should do is changing the input. (model and test directory)
