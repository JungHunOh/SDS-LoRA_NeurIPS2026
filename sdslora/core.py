''' SDS-LoRA (arXiv:2606.16454) on top of stock peft LoRA layers.

    Delta W = s (Q_B A + B Q_A^T) = s [Q_B  B] [A; Q_A^T]                          (Eq. 5)

    Q_A, Q_B are orthonormal bases of A^T and B kept as plain (non-parameter) tensors, so no
    gradient flows through them: dL/dA = s Q_B^T G, dL/dB = s G Q_A (Eq. 6). The LoRA layers'
    forward is replaced per instance, so the installed peft package is left untouched. '''
import math
import types

import torch
import torch.nn.functional as F
from peft.tuners.lora import Linear as LoraLinear
from transformers import TrainerCallback
from transformers.trainer_utils import SchedulerType

ADAPTER = 'default'


def lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoraLinear)]


def _sds_forward(self, x, *args, **kwargs):
    result = self.base_layer(x, *args, **kwargs)
    lora_A = self.lora_A[ADAPTER].weight
    lora_B = self.lora_B[ADAPTER].weight
    x = self.lora_dropout[ADAPTER](self._cast_input_dtype(x, lora_A.dtype))

    A = torch.cat([lora_A, self.sds_Q_A.T], dim=0)  # (2r, d_in)
    B = torch.cat([self.sds_Q_B, lora_B], dim=1)    # (d_out, 2r)
    return (result + F.linear(F.linear(x, A), B) * self.scaling[ADAPTER]).to(result.dtype)


@torch.no_grad()
def init_sdslora(model):
    ''' Eq. 8: with the warm-up update s B A = U S V^T, set
        A = S V^T / 2s, B = U S / 2s, Q_A = V, Q_B = U, so that s (Q_B A + B Q_A^T) = s B A.
        The SVD is taken of B A (= U (S/s) V^T), hence the 1/s is already folded into its singular values. '''
    for m in lora_modules(model):
        lora_A = m.lora_A[ADAPTER].weight
        lora_B = m.lora_B[ADAPTER].weight
        dtype = lora_A.dtype

        # exact rank-r SVD of B A through the r x r core: B = Q_b R_b, A^T = Q_a R_a, B A = Q_b (R_b R_a^T) Q_a^T
        Q_b, R_b = torch.linalg.qr(lora_B.float())
        Q_a, R_a = torch.linalg.qr(lora_A.float().T)
        u, s, vh = torch.linalg.svd(R_b @ R_a.T)
        U, V = Q_b @ u, Q_a @ vh.T

        lora_A.data = (0.5 * s[:, None] * V.T).contiguous().to(dtype)
        lora_B.data = (0.5 * U * s[None, :]).contiguous().to(dtype)
        m.sds_Q_A = V.contiguous().to(dtype)
        m.sds_Q_B = U.contiguous().to(dtype)
        m.forward = types.MethodType(_sds_forward, m)


@torch.no_grad()
def update_bases(model):
    ''' Q_A <- orth(A^T), Q_B <- orth(B) by QR, with column signs fixed so that diag(R) >= 0.
        Returns the fraction of layers whose bases were left unchanged by the norm check below. '''
    modules = lora_modules(model)
    skipped = 0
    for m in modules:
        lora_A = m.lora_A[ADAPTER].weight
        lora_B = m.lora_B[ADAPTER].weight

        # not described in the paper but used in the reported runs: skip a layer while any direction is near zero
        if not ((lora_B.norm(dim=0) > 1e-2).all() and (lora_A.norm(dim=1) > 1e-2).all()):
            skipped += 1
            continue

        Q_A, R_A = torch.linalg.qr(lora_A.float().T)
        Q_B, R_B = torch.linalg.qr(lora_B.float())
        m.sds_Q_A = (Q_A * torch.sign(torch.diag(R_A))).contiguous().to(lora_A.dtype)
        m.sds_Q_B = (Q_B * torch.sign(torch.diag(R_B))).contiguous().to(lora_A.dtype)

    return skipped / len(modules)


class SDSLoRACallback(TrainerCallback):
    ''' Re-orthonormalize the bases after optimizer step t when t mod ceil(k t / T_total) == 0 (Algorithm 1),
        i.e. every i-th step during the i-th of k equal phases. Logs `sds_skip_ratio`, the fraction of layers
        left unchanged at the most recent update. '''
    def __init__(self, model, k=5):
        self.model = model
        self.k = k
        self.skip_ratio = None

    def on_step_end(self, args, state, control, **kwargs):
        t = state.global_step
        if t % math.ceil(self.k * t / state.max_steps) == 0:
            self.skip_ratio = update_bases(self.model)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs and self.skip_ratio is not None:
            logs['sds_skip_ratio'] = self.skip_ratio


def warmup_and_init(trainer_cls, model, args, warmup_steps, **trainer_kwargs):
    ''' Warm-up stage: plain LoRA for `warmup_steps` optimizer steps at a constant learning rate, then Eq. 8.
        The main stage builds a new Trainer, so its optimizer state and LR schedule start from scratch. '''
    max_steps, lr_scheduler_type = args.max_steps, args.lr_scheduler_type
    args.max_steps, args.lr_scheduler_type = warmup_steps, SchedulerType.CONSTANT
    trainer_cls(model=model, args=args, **trainer_kwargs).train()
    args.max_steps, args.lr_scheduler_type = max_steps, lr_scheduler_type

    init_sdslora(model)
