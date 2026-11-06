# SDS-LoRA: Overcoming Anisotropic Gradient Scaling in Low-Rank Adaptation

Code for the commonsense-reasoning and math-reasoning experiments of
[SDS-LoRA (NeurIPS 2026, arXiv:2606.16454)](https://arxiv.org/abs/2606.16454). Each run fine-tunes a model with
rsLoRA (`rslora`) or SDS-LoRA (`sdslora`) and then evaluates it.

## Abstract

Low-Rank Adaptation (LoRA) enables efficient adaptation of large pretrained models to downstream tasks by
parameterizing weight updates with low-rank matrices. In this paper, we investigate the limitations of the LoRA
parameterization from a geometric perspective. Specifically, we show that when a full fine-tuning gradient is
backpropagated to the low-rank matrices, it undergoes anisotropic scaling driven by their singular values. We argue
that this phenomenon is undesirable because it distorts the full fine-tuning gradient by skewing it toward dominant
singular directions while suppressing others. Our analyses demonstrate that anisotropic gradient scaling reduces the
effective rank of the low-rank matrices' gradients and fails to provide the best possible alignment between the full
fine-tuning gradient and its low-rank approximation in LoRA for arbitrary gradients, thereby exacerbating the gap to
full fine-tuning. To address these limitations, we propose a new low-rank parameterization, SDS-LoRA, which
Structurally Decouples Singular values from the backward pass. Our method ensures that the full fine-tuning gradient
backpropagates only through the orthonormal bases of the low-rank matrices' subspaces, independent of their scales.
Convergence analysis demonstrates that while LoRA's convergence rate degrades with the condition number of the
low-rank matrices, that of SDS-LoRA remains independent of it. Experimental results across natural language and
vision benchmarks show that SDS-LoRA improves loss convergence and reduces the gap to full fine-tuning, significantly
enhancing adaptation performance.

## Setup

Tested with Python 3.13 and CUDA 12.4. Training runs in bf16, so it needs NVIDIA GPUs that support bf16.
The `torch==2.6.0` wheel from PyPI ships with the CUDA 12.4 runtime, so only an NVIDIA driver that supports
CUDA 12.4 is needed (no separate CUDA toolkit).

Run all commands in this README from the repository root.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
source .venv/bin/activate
```

`uv sync` downloads Python 3.13 if needed, creates `.venv`, and installs the exact versions in `uv.lock`.
Run `source .venv/bin/activate` again in each new shell before using the commands below.

Gemma-2B and LLaMA3-8B are gated on Hugging Face. Accept the license on the model page
([google/gemma-2b](https://huggingface.co/google/gemma-2b), [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B))
and log in with `hf auth login` before the first run.

Put the data under `data/`:

```
data/
├── commonsense/
│   ├── commonsense_170k.json          # LLM-Adapters: ft-training_set/commonsense_170k.json
│   └── eval/<dataset>/test.json       # LLM-Adapters: dataset/{boolq,piqa,social_i_qa,hellaswag,
│                                      #   winogrande,ARC-Challenge,ARC-Easy,openbookqa}/test.json
└── math/
    ├── MetaMathQA-395K.json           # huggingface.co/datasets/meta-math/MetaMathQA
    ├── GSM8K_test.jsonl               # MetaMath: data/test/GSM8K_test.jsonl
    └── MATH_test.jsonl                # MetaMath: data/test/MATH_test.jsonl
```

LLM-Adapters: <https://github.com/AGI-Edgerunners/LLM-Adapters>, MetaMath: <https://github.com/meta-math/MetaMath>.
To download everything:

```bash
LA=https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main
MM=https://raw.githubusercontent.com/meta-math/MetaMath/main/data/test
mkdir -p data/math
curl -fL --create-dirs -o data/commonsense/commonsense_170k.json $LA/ft-training_set/commonsense_170k.json
for d in boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa; do
  curl -fL --create-dirs -o data/commonsense/eval/$d/test.json $LA/dataset/$d/test.json
done
curl -fL -o data/math/GSM8K_test.jsonl $MM/GSM8K_test.jsonl
curl -fL -o data/math/MATH_test.jsonl $MM/MATH_test.jsonl
curl -fL -o data/math/MetaMathQA-395K.json \
  https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/main/MetaMathQA-395K.json
```

## Running

```bash
# one run per rank listed in the config (r: [8, 32])
python run.py configs/commonsense/gemma_sdslora.yaml --gpu 0,1,2,3
python run.py configs/math/llama3_rslora.yaml --gpu 0,1

# override any config value with dotted keys
python run.py configs/math/gemma_sdslora.yaml --gpu 0 r=32 seed=2 training_args.learning_rate=1e-4

# or launch a single run directly
torchrun --nproc_per_node=2 train.py configs/math/gemma_sdslora.yaml r=32
```

Add `--no_eval` to either command to skip evaluation.

Outputs go to `outputs/<task>/<model>_r<r>_<method>_seed<seed>/`. Each run directory holds:

| File | Contents |
|---|---|
| `log.txt` | Training log (stdout of `train.py`; progress bars and tracebacks go to the terminal) |
| `config.yaml` | Resolved config |
| `commonsense_eval.txt` | Accuracy on the 8 commonsense benchmarks and their average |
| `gsm8k_eval.txt`, `math_eval.txt` | GSM8K and MATH accuracy |
| `adapter_model.safetensors` | Saved LoRA weights (math only) |

`batch_size` is the global batch size. It is split across GPUs into `micro_batch_size` per GPU, with
gradient accumulation covering the rest. The effective batch size therefore does not depend on the number of GPUs,
as long as the GPU count divides `batch_size` (1, 2, 4 or 8 GPUs for the provided configs); `train.py` stops with an
error otherwise.

If a run fails, `run.py` exits with its exit code and does not start the remaining ranks.

## Configs

```
configs/
├── commonsense/_base.yaml        # settings shared by every commonsense run
├── commonsense/{gemma,llama3}_{rslora,sdslora}.yaml
├── math/_base.yaml
└── math/{gemma,llama3}_{rslora,sdslora}.yaml
```

A model/method config sets `defaults: _base.yaml` and overrides only what differs: the model, the method,
the per-GPU batch size, and the learning rate. `training_args` is passed to `transformers.TrainingArguments` as is.

| | Gemma-2B | LLaMA3-8B |
|---|---|---|
| Learning rate | 2e-4 | 1e-4 |
| Batch size / epochs / scheduler | 32 / 1 / cosine | 32 / 1 / cosine |
| LoRA scaling | rsLoRA, s = 4/√r | rsLoRA, s = 4/√r |
| SDS-LoRA | T = 10, k = 5 | T = 10, k = 5 |

## Code layout

```
sdslora/core.py          # SDS-LoRA
train.py                 # model / LoRA setup, warm-up, training, saving, evaluation
run.py                   # launcher (torchrun, rank sweep, logging)
config.py                # YAML loading with defaults and overrides
tasks/commonsense.py     # Commonsense-170K prompts and tokenization (LLM-Adapters)
tasks/commonsense_eval.py
tasks/math_reasoning.py  # MetaMathQA prompts and tokenization (MetaMath)
tasks/math_eval.py       # GSM8K / MATH answer extraction (MetaMath)
```

SDS-LoRA uses the stock `peft` LoRA layers. `sdslora/core.py` replaces each layer's `forward` per instance and
leaves the installed package unchanged. The mapping to the paper:

| Paper | Code |
|---|---|
| Eq. 5, ΔW = s(Q_B A + B Q_Aᵀ) | `_sds_forward` |
| Eq. 6, Q_A and Q_B treated as constants | Q_A and Q_B are plain tensors, not parameters |
| Eq. 8, initialization from the warm-up update | `init_sdslora` |
| Algorithm 1, warm-up stage | `warmup_and_init`: T steps of LoRA at a constant learning rate, then a fresh optimizer and LR schedule |
| Algorithm 1, QR update when t mod ⌈kt/T_total⌉ = 0 | `SDSLoRACallback` / `update_bases` |

Notes:
- `update_bases` leaves a layer's bases unchanged while any column of B or row of A has norm ≤ 1e-2.
  The training log reports the fraction of such layers at the most recent update as `sds_skip_ratio`.
- The saved adapter contains A and B only, not Q_A and Q_B. Evaluation runs on the in-memory model right after training.
  For LLaMA3 math runs, `peft` also saves `embed_tokens` and `lm_head` (about 2 GB): LLaMA3 has no pad token, so
  `tasks/math_reasoning.py` adds one and resizes the embeddings, and `peft` saves resized embedding layers.
- Commonsense runs use the HF `Trainer` seed 42 (`training_args.seed`), and `seed` for everything else.
  Math runs use `seed` for both.

## Acknowledgements

The data processing and evaluation code for commonsense reasoning is adapted from
[LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters). The code for math reasoning is adapted from
[MetaMath](https://github.com/meta-math/MetaMath).

## Citation

```bibtex
@inproceedings{oh2026sdslora,
  title     = {SDS-LoRA: Overcoming Anisotropic Gradient Scaling in Low-Rank Adaptation},
  author    = {Oh, Junghun and Baik, Sungyong and Lee, Kyoung Mu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE). Code adapted from LLM-Adapters and MetaMath (`tasks/`) remains under
their Apache-2.0 licenses.
