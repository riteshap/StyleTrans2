# Structured Inference-Time Control for Fine-Grained Text Style Transfer

## 1. Description

This repository contains the experimental code for **structured inference-time control of fine-grained text style transfer (TST)** using lightweight language models. The study investigates whether changing the structure and granularity of generation can improve adjacent-level style transfer without task-specific fine-tuning of the *generation model*.

The repository supports two complementary experimental settings:

- **Frozen-model inference-time control:** local **infilling** (`fill` in the code), local **replacement** (`replace`), sentence-level rewriting (`rewrite`), and sample-level rewriting (`base`). Local edits are combined into sentence- and sample-level candidates using style-score proxies and candidate reduction.
- **LoRA supervised fine-tuning (SFT):** task-specific pseudo-parallel data are generated with an external LLM API, validated and filtered, and then used to train LoRA adapters. The adapted models are evaluated using the same sample-level rewriting pipeline (`sft`).

Style transfer is assessed by **transfer success rate**, **semantic similarity**, and **fluency (perplexity, PPL)**. The repository also includes scripts for a separate inference-efficiency experiment and local-edit error analysis.

**Terminology:** “training-free” refers to *not updating the parameters of the text-generation model*. It does not mean that every auxiliary component is training-free: the Yelp sentiment evaluator uses a task-trained DistilBERT classifier.

**Execution note:** This is a research-code release, not a single-command application. Most scripts use configuration variables and local model paths defined near the bottom of the file; edit those settings before execution. The complete pipeline has not been packaged into one automated entry point.

## 2. Dataset Information

The repository contains prearranged, style-level-grouped JSON files for two English-language datasets:

| Dataset | Task | Levels | Included files |
| --- | --- | --- | --- |
| **Yelp Review Full** | Adjacent-level sentiment-intensity transfer | Five levels (`0`–`4`) | `data/yelp_train.json`, `data/yelp_test.json` |
| **CNN/DailyMail (CNN/DM)** | Adjacent-level readability-intensity transfer | Four levels (`0`–`3`) | `data/cnn_dm_train.json`, `data/cnn_dm_test.json` |

Each JSON file is a dictionary mapping a string-valued style level (for example, `"0"`) to a list of source texts. The supplied files have the following sample counts:

| File | Sample counts by level |
| --- | --- |
| `yelp_train.json` | 2,000 per level (five levels) |
| `yelp_test.json` | 1,000 per level (five levels) |
| `cnn_dm_train.json` | 2,000 per level (four levels) |
| `cnn_dm_test.json` | Level 0: 272; level 1: 1,494; level 2: 2,030; level 3: 202 |

The experimental scripts normally process a subset of these files (often up to 300 test samples per level); the actual number depends on data availability and the configured sample limit. The released CNN/DM test file has fewer than 300 samples at levels 0 and 3.

Source datasets: [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) and [CNN/DailyMail](https://github.com/abisee/cnn-dailymail). The provided JSON files are already grouped into style levels. **The original dataset acquisition and level-construction pipeline is not included** in this archive.

**Evaluation components:** Yelp style intensity is measured with a five-class DistilBERT model and a continuous probability-weighted sentiment score. CNN/DM style intensity is measured with Flesch Reading Ease (FRE). Semantic similarity uses `all-mpnet-base-v2`; fluency uses `Llama-3.2-3B-Instruct` to compute PPL. The Yelp classifier must be trained before running Yelp evaluation or SFT-data filtering.

## 3. Repository Structure

```text
.
├── data/
│   ├── yelp_train.json
│   ├── yelp_test.json
│   ├── cnn_dm_train.json
│   └── cnn_dm_test.json
├── model_pth/                    # Yelp classifier checkpoint after training
├── yelp_classification/
│   ├── pre_yelp4train.py         # Convert grouped Yelp JSON to [text, label] records
│   └── yelp_classification.py    # Train DistilBERT classifier; provide YelpScorer
├── sft/
│   ├── data_request.py          # Request pseudo-parallel training data via an LLM API
│   ├── merge_files.py           # Merge chunked API results
│   ├── data_prepare.py          # Parse/validate responses, filter, export SFT JSONL
│   ├── sft_data.py              # Source-grouped train/validation splitting
│   ├── sft_model.py             # LoRA training implementation
│   └── sft_train.py             # SFT training entry point
├── inference_control/
│   ├── constituency.py          # Stanza constituency parsing and editable spans
│   ├── prompt.py                # Infilling/replacement/rewrite prompt templates
│   ├── generation.py            # LLM generation helpers
│   ├── CandidateOut.py          # Source sentence records; raw local/sentence edits
│   ├── Evaluator.py             # Style and semantic scoring of edited sentences
│   ├── SentenceCandidate.py     # Compose/prune sentence-level candidates
│   ├── Combination.py           # Compose/select sample-level candidates
│   ├── SampleRewrite.py         # Full-sample rewriting: frozen model or LoRA-SFT
│   ├── SampleRewriteEvaluate.py # Add style/semantic scores to full rewrites
│   ├── Fluency.py               # Add PPL to local and full-rewrite outputs
│   └── Efficiency.py            # Independent timing experiment
└── evaluate/
    ├── ExperimentResultEvaluator.py  # Aggregate success, semantics and PPL
    └── ErrorAnalysis.py              # Inspect malformed local edits
```

**Generated intermediate files are not bundled.** Unless otherwise configured, the main intermediate outputs are written into `sft/`, `inference_control/`, or `evaluate/`, not into `data/`.

## 4. Usage Instructions

### 4.1 Initial configuration

1. Extract the repository and keep the directory structure shown above. Run commands from the **repository root**, except where a command explicitly changes into `sft/`.
2. Edit the hard-coded project roots, pretrained-model directories, checkpoint paths, task names, and model names in the scripts you intend to run. The released code contains machine-specific Windows (`F:\...`) and Linux (`/root/private_data/...`) paths; these are examples from the authors' environment, **not portable defaults**.
3. Make locally accessible copies of the required pretrained models and the Stanza English constituency-parser resources. Change `model_dir` in `inference_control/constituency.py`; also review the NLTK data path in `inference_control/Evaluator.py`. Several model loaders set `local_files_only=True`.
4. In the scripts' `__main__` blocks, select the task (`yelp` or `cnndm`), LLM (`gemma-4-12B-it` or `gemma-4-E2B-it`), and the appropriate input file. Keep `llm_dict` mappings consistent between stages. Output names use short model identifiers such as `ge12B` and `geE2B`.

The examples below show **execution order**, not plug-and-play commands: settings inside each file must be updated first. Scripts are run separately for each dataset/model configuration.

### 4.2 Train the Yelp sentiment classifier

The Yelp evaluator is required for Yelp experiments and for selecting Yelp SFT examples. First transform the level-grouped data into labeled pairs, then train DistilBERT:

```bash
python yelp_classification/pre_yelp4train.py
python yelp_classification/yelp_classification.py
```

- `pre_yelp4train.py` creates `data/yelp_train_list.json` and `data/yelp_test_list.json`.
- `yelp_classification.py` trains a five-class classifier based on `distilbert-base-uncased` and saves the best checkpoint at `model_pth/yelp_distilbert_best.pt`.
- Set the local DistilBERT model path and project root in the classifier script. Point all subsequent Yelp evaluators to the resulting checkpoint.

**Reproducibility note:** As released, the classifier script uses `yelp_test_list.json` for epoch-wise checkpoint selection. To use this classifier in an independently evaluated benchmark, create a separate development split rather than treating that same file as a fully untouched test set.

### 4.3 Prepare pseudo-parallel data and train LoRA-SFT (optional baseline)

This stage is required **only for LoRA-SFT experiments**; frozen-model inference-time-control experiments do not require an SFT adapter.

**Step A — Request training examples.** Edit `sft/data_request.py` to set `data_path`, `llm_name`, request index bounds, and **your own API key** (for example, read it from an environment variable instead of committing it to source control). Its current configuration uses a DeepSeek-compatible API and asks for three JSON fields: `weaker`, `same`, and `stronger`.

```bash
python sft/data_request.py
```

Outputs are written as chunked JSON files such as:

```text
sft/yelp_sft-traindata_-1_250.json
sft/cnndm_sft-traindata_-1_250.json
```

The index interval uses `min_sample_index < sample_id <= max_sample_index`; the default `-1, 250` covers indices 0–250. You can run additional non-overlapping chunks for a larger training set. API requests may incur external charges. The released `ds_api_key = "sk-"` is a placeholder and is not a usable credential.

**Step B — Merge chunk files.** Change `task` in `sft/merge_files.py`, then run it **from inside `sft/`** (it uses the current working directory):

```bash
cd sft
python merge_files.py
cd ..
```

This creates `sft/<task>_ori_traindata.json`.

**Step C — Validate, filter and format.** Set `task`, paths, and (for Yelp) the classifier checkpoint in `sft/data_prepare.py`:

```bash
python -m sft.data_prepare
```

This script performs three sequential operations:

1. Parse and validate the API's raw JSON responses and write `sft/<task>_formatted.json`.
2. Select style-valid generated examples (using the Yelp evaluator or CNN/DM FRE) and write `sft/<task>_selected.json`.
3. Convert the selected records into prompt/completion training examples at `sft/<task>_sft_formatted.jsonl`.

**Manual inspection may be required.** The format-validation stage expects valid JSON with string fields `weaker`, `same`, and `stronger`. If any record is malformed, `stage1_format()` reports the problematic record indices and **does not save the formatted file**. Inspect/correct the relevant `raw_output` fields in the merged input file, or regenerate those responses, and rerun the stage. Also inspect failed API requests and duplicate chunks before training. Existing intermediate output files can cause the scripts to skip processing; remove or rename stale outputs when intentionally rebuilding a stage.

**Step D — Train the adapter.** Edit `sft/sft_train.py` for your model path, dataset, output directory and hyperparameters. In particular, replace the example `output_dir="/data"` used for training/validation split files with a writable directory in your environment. Run the training entry point from `sft/` so its relative `outputs/` directory matches the inference code's expected adapter location:

```bash
cd sft
python sft_train.py
cd ..
```

The training entry point uses `SFTDataModule` to form a source-grouped training/validation split and `SFTModel` to train a LoRA adapter. The default configuration specifies three epochs, LoRA rank 16 and alpha 32. The saved adapter is expected at `sft/outputs/<task>_<model-id>_sft/final_adapter/` when `outputs/` is relative to `sft/`. Adjust the adapter path in `inference_control/SampleRewrite.py` if your save location differs.

### 4.4 Structured inference-time control: local editing and sentence rewriting

This workflow covers **local infilling (`fill`), local replacement (`replace`), and sentence-level rewriting (`rewrite`)**, using a frozen generation model. Set the same task, model, evaluator, and paths in every stage.

**Stage 1 — Generate source records and candidate edits.** Edit `inference_control/CandidateOut.py`. Near the bottom of the released file, `source_main()` is enabled but `main()` is **commented out**. Enable the latter to generate candidates:

```python
cg_class.source_main(data_file_path)  # Source sentence segmentation/metadata
cg_class.main(data_file_path, total_num=300, fill_batch=16)  # Edit generation
```

Then run:

```bash
python inference_control/CandidateOut.py
```

Outputs:

```text
inference_control/<task>_source_sentence.json
inference_control/<task>_<model-id>_edits.json
```

The original `data/yelp_test.json` or `data/cnn_dm_test.json` remains an **input**; it is not overwritten by this script. The edits file contains generated local units and sentence-level rewriting candidates together with identifiers, directions and source-sentence information.

**Stages 2–5 — Score, compose, and evaluate.** After configuring each file's `__main__` block, run in order:

```bash
python inference_control/Evaluator.py
python inference_control/SentenceCandidate.py
python inference_control/Combination.py
python inference_control/Fluency.py
```

| Script | Main operation | Output |
| --- | --- | --- |
| `Evaluator.py` | Reconstruct and score edited sentences; apply sentence-level semantic filtering | `<task>_<model-id>_evaluated.json` |
| `SentenceCandidate.py` | Combine local edits and prune sentence-level candidate states (typically up to 20 retained candidates) | `<task>_<model-id>_sentence-cand.pkl` |
| `Combination.py` | Combine sentence states into full-sample candidates and calculate style and semantic scores | `<task>_<model-id>_combination.json` |
| `Fluency.py` (`main()`) | Calculate source/output PPL with the fluency model | `<task>_<model-id>_results.json` |

These output files are created in `inference_control/`. The intermediate `*_combination.json` **does not contain PPL**; `Fluency.py` adds fluency results in the final `*_results.json`. The sentence candidates are serialized in a pickle file, which should only be loaded from a trusted source.

### 4.5 Full-sample rewriting: frozen model and LoRA-SFT

Both full-sample baselines use the same two-stage rewriting-and-evaluation code. For each run, edit `mode` in `inference_control/SampleRewrite.py` and `inference_control/SampleRewriteEvaluate.py` to either `base` (frozen model) or `sft` (LoRA-adapted model), and configure the dataset/model paths.

```bash
python inference_control/SampleRewrite.py
python inference_control/SampleRewriteEvaluate.py
```

This produces:

```text
inference_control/<task>_<model-id>_<mode>_sample_rewrite.json
inference_control/<task>_<model-id>_<mode>_sample_rewrite_v2.json
```

The first file stores generated rewrite candidates; the second adds style and semantic similarity scores. Next configure `mode` in `inference_control/Fluency.py` and run it to add PPL:

```bash
python inference_control/Fluency.py
```

Final output:

```text
inference_control/<task>_<model-id>_<mode>_sample_rewrite_results.json
```

Repeat for `mode = "base"` and `mode = "sft"` if reproducing both baselines. The SFT mode must point to the correct LoRA adapter directory.

**Important sample-limit setting:** In the released `SampleRewrite.py`, the per-style-level limit is **300 samples for `base` but 100 for `sft`**. Confirm and harmonize these limits if you intend to compare the two modes on exactly the same examples. Candidate generation defaults to 20 output candidates per rewrite request; the structured local pipeline can retain fewer than 20 distinct final candidates in some cases.

### 4.6 Aggregate evaluation and error analysis

Once the final files containing style, semantic, and PPL results have been generated, edit the task, model and paths in `evaluate/ExperimentResultEvaluator.py`:

```bash
python evaluate/ExperimentResultEvaluator.py
```

The evaluator summarizes **transfer success rate**, **semantic preservation**, and **fluency (PPL)**. Its `main()` handles the local/sentence-level outputs; `sample_rewrite_main()` handles full-sample rewriting, and `sample_rewrite_n_analysis()` supports candidate-budget analysis. Set its sample-rewriting mode (`base` or `sft`) to match the file being analyzed. Results are printed, and selected records for the local/sentence-level analysis may also be written to `evaluate/<task>_<model-id>_results.json`.

For abnormal-format analysis of local edits, configure and run:

```bash
python evaluate/ErrorAnalysis.py
```

It examines raw local-edit outputs and final selected records, including unwanted characters or formatting artifacts; it may write `evaluate/<task>_<model-id>_results_errors.json`.

### 4.7 Independent inference-efficiency experiment

`inference_control/Efficiency.py` is a **separate timing script** that executes a small configurable sample-level pipeline for one mode (`fill`, `replace`, or `rewrite`). It is not a required prerequisite for the main candidate-generation pipeline.

Edit `test_level`, `test_sample_num`, `test_mode`, `test_direction`, model paths, and task settings inside `main()`; note that the released script contains multiple path assignments and the later assignments take precedence. Run:

```bash
python inference_control/Efficiency.py
```

This script reports elapsed time for the configured trial. Full-sample rewriting time can be obtained from its separate generation workflow; `Efficiency.py` is specifically organized around the local/sentence-control modes.

## 5. Requirements

### Software

- **Python 3.12 or later** is recommended for the syntax used in this release.
- **PyTorch** and **Transformers** with support for the selected local models.
- **PEFT**, **TRL**, **Datasets**, and **Accelerate** for LoRA-SFT.
- **Sentence-Transformers** for semantic similarity.
- **Stanza** and **NLTK** for constituency-based editable-unit extraction.
- **textstat** for Flesch Reading Ease.
- **OpenAI Python SDK** (`openai`) for the external, OpenAI-compatible SFT data-generation API.
- **NumPy** and **tqdm** for statistics and training utilities.
- Optional: **bitsandbytes** if you enable the 4-bit training option in `sft_model.py`.

An example installation command (use package versions compatible with your CUDA/PyTorch environment):

```bash
pip install torch transformers accelerate peft trl datasets sentence-transformers stanza nltk textstat openai numpy tqdm
```

### Models and hardware

- Generator: local files for the selected **Gemma-4-12B-it** or **Gemma-4-E2B-it** model (as configured in the scripts).
- Yelp style evaluator: **distilbert-base-uncased** plus the trained checkpoint `model_pth/yelp_distilbert_best.pt`.
- Semantic evaluator: **sentence-transformers/all-mpnet-base-v2**.
- Fluency evaluator: **Llama-3.2-3B-Instruct**.
- Stanza English `tokenize`, `pos` and `constituency` resources; any required local NLTK data.
- A CUDA-capable GPU with sufficient memory is strongly recommended; several scripts explicitly request CUDA, automatic device mapping and/or `bfloat16`. The study used an NVIDIA A800, but the code does not enforce that specific GPU.
- For SFT pseudo-data generation: a valid API key, access to the configured external model endpoint and an internet connection for requests.
