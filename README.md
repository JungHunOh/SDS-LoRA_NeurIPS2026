# SDS-LoRA: Overcoming Anisotropic Gradient Scaling in Low-Rank Adaptation

Code for the commonsense-reasoning and math-reasoning experiments of
[SDS-LoRA (arXiv:2606.16454)](https://arxiv.org/abs/2606.16454). Each run fine-tunes a model with
LoRA (`base`) or SDS-LoRA (`sdslora`) and then evaluates it.

## Setup

Tested with Python 3.13 and CUDA 12.4.

```bash
pip install -r requirements.txt
```

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

## Running

```bash
# one run per rank listed in the config (r: [8, 32])
python run.py configs/commonsense/gemma_sdslora.yaml --gpu 0,1,2,3
python run.py configs/math/llama3_base.yaml --gpu 0,1

# override any config value with dotted keys
python run.py configs/math/gemma_sdslora.yaml --gpu 0 r=32 seed=2 training_args.learning_rate=1e-4

# or launch a single run directly
torchrun --nproc_per_node=2 train.py configs/math/gemma_sdslora.yaml r=32
```

Add `--no_eval` to either command to skip evaluation.

Outputs go to `outputs/<task>/<model>_r<r>_<method>_seed<seed>/`. Each run directory holds:

| File | Contents |
|---|---|
| `log.txt` | Training log |
| `config.yaml` | Resolved config |
| `commonsense_eval.txt` | Accuracy on the 8 commonsense benchmarks and their average |
| `gsm8k_eval.txt`, `math_eval.txt` | GSM8K and MATH accuracy |
| `adapter_model.safetensors` | Saved LoRA weights (math only) |

`batch_size` is the global batch size. It is split across GPUs into `micro_batch_size` per GPU, with
gradient accumulation covering the rest. The effective batch size therefore does not depend on the number of GPUs.

## Configs

```
configs/
├── commonsense/_base.yaml        # settings shared by every commonsense run
├── commonsense/{gemma,llama3}_{base,sdslora}.yaml
├── math/_base.yaml
└── math/{gemma,llama3}_{base,sdslora}.yaml
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
- The saved adapter contains A and B only. Evaluation runs on the in-memory model right after training.
- Commonsense runs use the HF `Trainer` seed 42 (`training_args.seed`), and `seed` for everything else.
  Math runs use `seed` for both.

## Acknowledgements

The data processing and evaluation code for commonsense reasoning is adapted from
[LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters). The code for math reasoning is adapted from
[MetaMath](https://github.com/meta-math/MetaMath).

## Citation

```bibtex
@article{oh2026sdslora,
  title   = {SDS-LoRA: Overcoming Anisotropic Gradient Scaling in Low-Rank Adaptation},
  author  = {Oh, Junghun and Baik, Sungyong and Lee, Kyoung Mu},
  journal = {arXiv preprint arXiv:2606.16454},
  year    = {2026}
}
```

## License

MIT License. See [LICENSE](LICENSE). Code adapted from LLM-Adapters and MetaMath (`tasks/`) remains under
their Apache-2.0 licenses.
