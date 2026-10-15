''' Fine-tune with rsLoRA (`method: rslora`) or SDS-LoRA (`method: sdslora`), then evaluate on rank 0.

    torchrun --nproc_per_node=<num_gpus> train.py <config.yaml> [key=value ...] [--no_eval]

    run.py wraps this and sweeps over the ranks listed in the config. '''
import argparse
import os

import torch
import transformers
import yaml
from peft import LoraConfig, get_peft_model

from config import load_config, output_dir_of
from sdslora import SDSLoRACallback, warmup_and_init
from tasks import get_task


def train(cfg):
    task = get_task(cfg['task'])
    assert cfg['method'] in ('rslora', 'sdslora'), cfg['method']
    assert isinstance(cfg['r'], int), "pick a single rank, e.g. r=32 (run.py sweeps over a list)"
    output_dir = output_dir_of(cfg)

    transformers.set_seed(cfg['seed'])

    # batch_size is the global batch; split it over processes, then gradient accumulation
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    micro_batch_size = min(cfg['micro_batch_size'], cfg['batch_size'] // world_size)
    grad_accum = cfg['batch_size'] // micro_batch_size // world_size

    model = transformers.AutoModelForCausalLM.from_pretrained(cfg['base_model'], torch_dtype=torch.bfloat16)
    tokenizer = task.load_tokenizer(cfg, model)
    train_data, data_collator = task.load_train_data(cfg, tokenizer)

    model = get_peft_model(model, LoraConfig(
        r=cfg['r'],
        lora_alpha=cfg['lora_alpha'],
        target_modules=cfg['target_modules'],
        lora_dropout=cfg['lora_dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    ))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

    training_args = {'seed': cfg['seed'], **cfg['training_args']}
    args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        **training_args,
    )
    trainer_kwargs = dict(train_dataset=train_data, data_collator=data_collator, processing_class=tokenizer)

    callbacks = []
    if cfg['method'] == 'sdslora':
        warmup_and_init(task.Trainer, model, args, cfg['sds_warmup_steps'], **trainer_kwargs)
        callbacks.append(SDSLoRACallback(model, k=cfg['sds_k']))

    if cfg.get('disable_kv_cache', False):
        model.config.use_cache = False
    trainer = task.Trainer(model=model, args=args, callbacks=callbacks, **trainer_kwargs)
    trainer.train()
    return model, tokenizer, trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('overrides', nargs='*', help='key=value, e.g. r=8 training_args.learning_rate=1e-4')
    parser.add_argument('--no_eval', action='store_true')
    cli = parser.parse_intermixed_args()

    cfg = load_config(cli.config, cli.overrides)
    model, tokenizer, trainer = train(cfg)
    if not trainer.is_world_process_zero():
        return

    output_dir = output_dir_of(cfg)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    if cfg.get('save_adapter', False):
        # SDS-LoRA's Q_A / Q_B are not part of the saved adapter
        trainer.save_state()
        for param in model.parameters():
            param.data = param.data.contiguous()
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    torch.cuda.empty_cache()
    if not cli.no_eval:
        get_task(cfg['task']).evaluate(model, tokenizer, cfg, output_dir)


if __name__ == "__main__":
    main()
