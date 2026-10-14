''' YAML experiment configs. A config may name a `defaults` file (relative to itself) whose values it overrides;
    command-line overrides use dotted keys, e.g. `r=8 training_args.learning_rate=1e-4`. '''
import os

import yaml


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _parse(value):
    value = yaml.safe_load(value)
    if isinstance(value, str):  # YAML 1.1 reads '1e-4' as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if 'defaults' in cfg:
        cfg = _merge(load_config(os.path.join(os.path.dirname(path), cfg.pop('defaults'))), cfg)

    for item in overrides:
        key, value = item.split('=', 1)
        *parents, leaf = key.split('.')
        node = cfg
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = _parse(value)
    return cfg


def output_dir_of(cfg):
    if cfg.get('output_dir'):
        return cfg['output_dir']
    return os.path.join('outputs', cfg['task'], f"{cfg['model']}_r{cfg['r']}_{cfg['method']}_seed{cfg['seed']}")
