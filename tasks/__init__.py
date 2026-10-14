''' Each task module provides load_tokenizer(cfg, model), load_train_data(cfg, tokenizer) -> (dataset, collator),
    Trainer (the transformers.Trainer class to use) and evaluate(model, tokenizer, cfg, output_dir). '''
import importlib

TASKS = {'commonsense': 'tasks.commonsense', 'math': 'tasks.math_reasoning'}


def get_task(name):
    return importlib.import_module(TASKS[name])
