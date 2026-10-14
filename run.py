''' Run train.py on the given GPUs, once per rank listed in the config. stdout goes to <output_dir>/log.txt.

    python run.py configs/math/gemma_sdslora.yaml --gpu 0,1 [key=value ...] '''
import argparse
import os
import shlex
import socket
import subprocess
import sys

from config import load_config, output_dir_of

ROOT = os.path.dirname(os.path.abspath(__file__))


def free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config')
    parser.add_argument('overrides', nargs='*', help='key=value, e.g. seed=2 training_args.learning_rate=1e-4')
    parser.add_argument('--gpu', default='0', help='comma-separated GPU ids')
    parser.add_argument('--no_eval', action='store_true')
    cli = parser.parse_intermixed_args()
    os.chdir(ROOT)

    cfg = load_config(cli.config, cli.overrides)
    ranks = cfg['r'] if isinstance(cfg['r'], list) else [cfg['r']]
    for r in ranks:
        overrides = cli.overrides + [f'r={r}']
        output_dir = output_dir_of(load_config(cli.config, overrides))
        os.makedirs(output_dir, exist_ok=True)

        cmd = [sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={len(cli.gpu.split(","))}',
               f'--master_port={free_port()}', 'train.py', cli.config, *overrides]
        if cli.no_eval:
            cmd.append('--no_eval')
        cmd = f"{shlex.join(cmd)} | tee {shlex.quote(os.path.join(output_dir, 'log.txt'))}"
        print(f"[run.py] {cmd}", flush=True)
        subprocess.run(cmd, shell=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': cli.gpu})


if __name__ == '__main__':
    main()
