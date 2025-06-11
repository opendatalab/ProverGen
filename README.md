<div align="center">
<!-- <img style="width: 50%;" src="assets/head-image.png"> -->
<!-- <h2 align="center"><img style="height: 40px;" src="https://github.com/opendatalab/LOKI/blob/27f9fa838ee344798e210ee00fa70ab1b32ef6ae/static/img/icons/loki.png"> $\LARGE\textbf{\textsf{{\color[rgb]{1.0, 0.7, 0.0}L}{\color[rgb]{1.0, 0.6, 0.0}O}{\color[rgb]{1.0, 0.5, 0.0}K}{\color[rgb]{1.0, 0.4, 0.0}I}}}{\color[rgb]{0,0,0}}$ 
 A Comprehensive Synthetic Data Detection Benchmark using Large Multimodal Models</h2> -->
 <h2 align="center">Large Language Models Meet Symbolic Provers for Logical Reasoning Evaluation</h2>

<a href="https://openreview.net/forum?id=C25SgeXWjE" target="_blank"><img src="https://img.shields.io/badge/ICLR-ProverGen-red?style=badge&logo=arXiv" alt="Paper PDF" height="25"></a>
<a href='https://huggingface.co/datasets/opendatalab/ProverQA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ProverQA-yellow' height="25"></a>
<a href='https://opendatalab.com/OpenDataLab/ProverQA'><img src='https://img.shields.io/badge/OpenDataLab-ProverQA-green' height="25"></a>
</div>

# ProverGen: A Framework for First-Order Logic Reasoning Dataset Generation

<div style="text-align: center;">
    <img src="./framework.png" alt="The general framework of ProverGen" width="90%" />
</div>

## Overview

**ProverGen** is a novel framework that synergizes the generative strengths of Large Language Models (LLMs) with the rigor and precision of symbolic provers to create scalable, diverse, and high-quality First-Order Logic (FOL) reasoning datasets. 

Using this framework, we have created **ProverQA**, a challenging FOL reasoning benchmark consisting of 1,500 evaluation instances and 5,000 training instances across three difficulty levels.

### Key Features

- **🎯 Scalable Generation**: Automated framework enabling expansion with minimal manual intervention
- **🌍 Natural & Diverse Language**: Wide range of natural language expressions reflecting real-world linguistic variability  
- **🔬 Symbolic Validation**: Formal symbolic structures validated through Prover9 symbolic prover
- **🔗 Faithful Reasoning Chains**: Transparent intermediate reasoning steps in both symbolic and natural language formats
- **📊 Complete FOL Coverage**: All seven First-Order Logic relationships (∧, ∨, ¬, →, ⊕, ∀, ∃)

## Table of Contents

- [ProverGen: A Framework for First-Order Logic Reasoning Dataset Generation](#provergen-a-framework-for-first-order-logic-reasoning-dataset-generation)
  - [Overview](#overview)
    - [Key Features](#key-features)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Install Required Packages](#install-required-packages)
    - [Install Prover9](#install-prover9)
  - [Dataset: ProverQA](#dataset-proverqa)
  - [Generation Pipeline](#generation-pipeline)
    - [Step 1: Logic Skeleton Generation 🧭](#step-1-logic-skeleton-generation-)
    - [Step 2: Logic Skeleton Translation 📝](#step-2-logic-skeleton-translation-)
    - [Step 3: FOL Problem Generation 📚](#step-3-fol-problem-generation-)
  - [Evaluation ⚖️](#evaluation-️)
  - [Benchmark Results](#benchmark-results)
  - [Citation](#citation)
  - [License and Ethics](#license-and-ethics)
  - [Contact](#contact)

## Installation

### Install Required Packages
```bash
conda create -n provergen python=3.8
conda activate provergen
git clone https://github.com/opendatalab/ProverGen
cd ./ProverGen
pip install -r requirements.txt
```

### Install Prover9
The ProverGen framework leverages [Prover9](https://www.cs.unm.edu/~mccune/mace4/) for logic skeleton generation. Download from their [website](https://www.cs.unm.edu/~mccune/mace4/download/) (select `LADR-2009-11A.tar.gz`). For installation instructions, refer to the [guide](https://www.cs.unm.edu/~mccune/mace4/manual/2009-11A/).

⚠️ **Installation Notes**

**Linux systems:** Since Prover9 is outdated, you might encounter issues with `make all`. Navigate to `LADR-2009-11A/provers.src/Makefile` and move all `-lm` flags to the end of each line:

```makefile
# Before
$(CC) $(CFLAGS) -lm -o newsax newsax.o $(OBJECTS) ../ladr/libladr.a

# After  
$(CC) $(CFLAGS) -o newsax newsax.o $(OBJECTS) ../ladr/libladr.a -lm
```

**macOS systems:** You may encounter implicit declaration errors. Fix by:

1. In `LADR-2009-11A/mace4.src/select.c: line 236`:
```c
// Before
int select_concentric_band(min_id, max_id, max_constrained)
// After
int select_concentric_band(int min_id, int max_id, int max_constrained)
```

2. In `LADR-2009-11A/mace4.src/msearch.c: line 850`:
```c
// Before
int next_domain_size(n)
// After
int next_domain_size(int n)
```

After installation, update the binary locations in `logic_skeleton_generator.py` at line 20.

## Dataset: ProverQA

The **ProverQA** dataset created using this framework is available on:
- 🤗 **Hugging Face**: [opendatalab/ProverQA](https://huggingface.co/datasets/opendatalab/ProverQA)
- 🌐 **OpenDataLab**: [ProverQA](https://opendatalab.com/maren/ProverQA)

**Dataset Structure:**
- **Development Set**: 1,500 instances (500 easy, 500 medium, 500 hard)
- **Training Set**: 5,000 instances with data augmentation
- **Difficulty Levels**: Easy (1-2 steps), Medium (3-5 steps), Hard (6-9 steps)

## Generation Pipeline

The generation process consists of three main steps:

### Step 1: Logic Skeleton Generation 🧭

Generate the logical structure using symbolic provers with a novel top-down approach.

```bash
python3 logic_skeleton_generator.py --mode easy --num 500 --output_dir outputs/logic_data
```

**Parameters:**
- `mode`: Difficulty level (`easy`, `medium`, `hard`)  
- `num`: Number of logic skeletons to generate
- `output_dir`: Output directory

The script also allows customization of the distribution of answers ([True, False, Uncertain]) and the proportion of composite conclusions. 

<details><summary>Here are the relevant parameters:</summary><p>

- `goal_value_probs`: Distribution of [True, False, Uncertain] (e.g., [0.4, 0.3, 0.3]).
- `rule_candidate_path`: Path to the rule pool file.
- `rule_as_goal_proportion`: Proportion of fact vs. rule conclusions (e.g., [0.7, 0.3]).
- `fact_num_threshold`: If the fact pool size surpasses this threshold, there's a chance the fact will be provided directly.
- `fact_num_prob`: Probability of directly providing a fact.
</p></details>

### Step 2: Logic Skeleton Translation 📝

Convert logic expressions into natural language using LLMs.

```bash
python3 logic_skeleton_translator.py \
    --model_name meta-llama/Meta-Llama-3.1-70B-Instruct \
    --data_dir outputs/logic_data \
    --num 100 --start 0 --end 100 \
    --output_dir outputs/translated_data \
    --mode hard \
    --base_url localhost:6417 --api_key EMPTY
```

**Parameters:**
- `model_name`: LLM for translation
- `data_dir`: Path to the logic skeleton files produced in Step 1.
- `num`: Total number of logic skeletons
- `start`/`end`: Index range for processing
- `output_dir`: Directory to store the output file.
- `mode`: Difficulty level
- `base_url`/`api_key`: For local models

### Step 3: FOL Problem Generation 📚

Generate complete FOL problems with optional data augmentation.

```bash
python3 fol_problem_generator.py \
    --model_name meta-llama/Meta-Llama-3.1-70B-Instruct
    --filepath outputs/translated_data/hard-100-0_100.json \
    --start 0 --end 100 \
    --output_dir outputs/final_data \
    --mode normal_generation \
```

**Parameters:**
- `model_name`: LLM for generation.
- `filepath`: Path to the translated files produced in Step 2.
- `start`/`end`: Index range for processing
- `output_dir`: Directory to store the output file.
- `mode`: Generation mode (`normal_generation`, `step_augment`, `uncertain_augment`)
- `base_url`/`api_key`: For local models
- `noise1`/`noise2`: Control different types of distractions

## Evaluation ⚖️

Evaluate LLM performance on the generated datasets:

```bash
python3 evaluation.py \
    --model_name meta-llama/Meta-Llama-3.1-70B-Instruct \
    --dataset_name ProverQA --split easy \
    --output_dir result/ \
    --mode Direct \
    --start 0 --end 500
    --base_url http://localhost:6417/v1 --api_key EMPTY \
```

**Parameters:**
- `model_name`: LLM for evaluation.
- `dataset_name`: Dataset to evaluate (`ProverQA`, `FOLIO`, `ProntoQA`, `ProofWriter`)
- `split`: Subset (`easy`, `medium`, `hard` for ProverQA; `dev` for others)
- `output_dir`: Directory to store the output file.
- `mode`: Evaluation strategy (`Direct` or `CoT`)
- `start`/`end`: Index range for evaluation.
- `base_url`/`api_key`: For local models
- `trained_model`: Path to fine-tuned model (optional)
  

Compute metrics using `metric.py` on the generated result files.

## Benchmark Results

State-of-the-art models show significant room for improvement on ProverQA:

| Model | Easy | Medium | Hard |
|-------|------|--------|------|
| **Proprietary Models** | | | |
| GPT-4o (CoT) | 94.2% | 79.4% | 50.0% |
| Claude-3.5-Sonnet (CoT) | 95.2% | 83.6% | 56.4% |
| o1-preview-2024-09-12 | 89.8% | 78.8% | 66.2% |
| **Open Models** | | | |
| Llama3.1-70B (CoT) | 90.4% | 73.2% | 46.8% |
| Llama3.1-8B (CoT) | 75.6% | 46.6% | 33.6% |
| Mistral-Large (CoT) | 92.6% | 75.8% | 52.2% |
| DeepSeek-R1 | 91.8% | 78.4% | 66.6% |

Key observations:
- 📉 **Challenging Hard Subset**: Even top models struggle with 6-9 step reasoning
- 🧠 **Reasoning Models Excel**: O1 and DeepSeek-R1 show strong performance on complex problems  
- 🔗 **CoT Helps**: Chain-of-thought prompting provides significant improvements
- 📊 **Fine-grained Difficulty**: Clear performance degradation across difficulty levels

## Citation

If you use ProverGen or ProverQA in your research, please cite:

```bibtex
@inproceedings{
qi2025large,
title={Large Language Models Meet Symbolic Provers for Logical Reasoning Evaluation},
author={Chengwen Qi and Ren Ma and Bowen Li and He Du and Binyuan Hui and Jinwang Wu and Yuanjun Laili and Conghui He},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=C25SgeXWjE}
}
```

## License and Ethics

- ✅ **Open Source**: Framework and dataset are publicly available
- 📚 **Compliance**: Dataset sources follow public repository licenses (MIT for names, WordNet for keywords)
- 🤖 **No Human Data**: No human participants involved in data collection
- 🛡️ **Safety**: Dataset contains no harmful or biased content
- 🎓 **Academic Purpose**: Designed for research in advancing logical reasoning capabilities

## Contact

For questions, issues, or collaborations:

- **Chengwen Qi**: chengwen_qi@buaa.edu.cn
- **Ren Ma**: maren@pjlab.org.cn  
- **Bowen Li**: libowen@pjlab.org.cn

**GitHub Issues**: [https://github.com/opendatalab/ProverGen/issues](https://github.com/opendatalab/ProverGen/issues)
