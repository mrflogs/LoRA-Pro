from typing import Tuple, Union

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from typing import Callable, Iterable, Tuple, Optional, Dict, List

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch import Tensor
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
from torch._utils import is_compiling
from torch.optim.optimizer import _use_grad_for_differentiable

import peft
from transformers import TrainingArguments, Trainer

from collections import OrderedDict
import math

from scipy.linalg import solve_sylvester

import wandb
import torch.distributed as dist
import logging

def _dispatch_sqrt(
    x: float,
):  # float annotation is needed because of torchscript type inference
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return math.sqrt(x)

def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and is_compiling():
        return x
    else:
        return x.item() if isinstance(x, torch.Tensor) else x

def _get_scalar_dtype():
    return (
        torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    )

def _warmup_lr(base_lr: float, warmup_length: int, step_idx: int):
    return base_lr * (step_idx + 1) / warmup_length


def _cos_lr(base_lr: float, max_steps: int, step_idx: int):
    lr = 0.5 * (1 + np.cos(np.pi * step_idx / max_steps)) * base_lr
    return lr

def solve_sylvester(A, B, C, X=None):
    ''' From the answer here: 
        https://stackoverflow.com/questions/73713072/solving-sylvester-equations-in-pytorch
    '''
    if A.dtype is torch.bfloat16:
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        C = C.to(torch.float32)
    B = -B
    m = B.shape[-1];
    n = A.shape[-1];
    R, U = torch.linalg.eig(A)
    S, V = torch.linalg.eig(B)
    F = torch.linalg.solve(U, (C + 0j) @ V)
    W = R[..., :, None] - S[..., None, :]
    Y = F / W
    X = U[...,:n,:n] @ Y[...,:n,:m] @ torch.linalg.inv(V)[...,:m,:m]
    return X.real if all(torch.isreal(x.flatten()[0]) 
                for x in [A, B, C]) else X

class LinearWarmup:
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lrs: Union[float, Tuple[float]],
        warmup_steps: int,
        max_steps: int,
    ):
        super().__init__()
        self.optimizer = optimizer
        if isinstance(base_lrs, (float, int)):
            base_lrs = tuple(base_lrs for _ in optimizer.param_groups)
        self.base_lrs = base_lrs
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lrs(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, step_idx: int = 0):
        warmup_length = self.warmup_steps
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            
            if step_idx < warmup_length:
                lr = base_lr * float(step_idx) / float(max(1, warmup_length))
            else:
                lr = base_lr * max(0.0, float(self.max_steps - step_idx) / float(max(1, self.max_steps - warmup_length)))
            param_group["lr"] = lr  # assign learning rate

        self._last_lr = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]

class CosineAnnealingWithWarmup:
    R"""
    a `max_steps`-step cosine annealing learning rate schedule with `warmup_steps` warm-up steps.
    The `step(step_idx)` method should be called every update step.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lrs: Union[float, Tuple[float]],
        warmup_steps: int,
        max_steps: int,
    ):
        super().__init__()
        self.optimizer = optimizer
        if isinstance(base_lrs, (float, int)):
            base_lrs = tuple(base_lrs for _ in optimizer.param_groups)
        self.base_lrs = base_lrs
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lrs(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, step_idx: int = 0):
        warmup_length = self.warmup_steps
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if step_idx < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step_idx)
            else:
                lr = _cos_lr(
                    base_lr, self.max_steps - warmup_length, step_idx - warmup_length
                )
            param_group["lr"] = lr  # assign learning rate

        self._last_lr = [
            param_group["lr"] for param_group in self.optimizer.param_groups
        ]
        
        
class AdamW(Optimizer):
    def __init__(
        self,
        named_params,
        lr: Union[float, Tensor] = 1e-3,
        scaling_factor: float = 2.,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        maximize: bool = False, 
        differentiable: bool = False,
        mode: str = "efficient",
        X_mode: str = "sylvester",
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not mode in ["full", "efficient"]:
            raise ValueError(f"Invalid mode value: {mode}, mode should be in ['full', 'efficient']")
        if not X_mode in ["zero", "sylvester", "symmetry"]:
            raise ValueError(f"Invalid mode value: {X_mode}, mode should be in ['zero', 'sylvester', 'symmetry']")

        names = []
        params = []
        for n, p in named_params:
            names.append(n)
            params.append(p)
            
        self.mode = mode
        self.X_mode = X_mode
        self.step_ = 0
        defaults = dict(
            lr=lr,
            names=names,
            scaling_factor=scaling_factor,
            X_mode=X_mode,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def step_full(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            scaling_factor = group["scaling_factor"]

            param_list = []
            param_dict = dict(zip(group["names"], group["params"]))
            for n, p in param_dict.items():
                if p.grad is None:
                    continue
                if 'lora' in n:
                    param_list.append(p)
                    if len(param_list) == 2:
                        base_name = n[: n.find('lora')] 
                        name = base_name + 'lora'
                        weight_name = base_name + "base_layer.weight"
                        size = (param_list[1].shape[0], param_list[0].shape[1])
                    elif len(param_list) == 1:
                        continue
                else:
                    name = n
                    size = p.shape
                
                state = self.state[name]
            
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros(size).to(p.device).to(p.dtype).to(torch.bfloat16)

                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros(size).to(p.device).to(p.dtype).to(torch.bfloat16)

                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros(size).to(p.device).to(p.dtype).to(torch.bfloat16)

                # step
                if len(param_list) == 2:
                    
                    # list_param = [A, B]
                    A = param_list[0]
                    B = param_list[1]
                    grad_A_orin = A.grad
                    grad_B_orin = B.grad
                    
                    # projection
                    delta = 1e-8
 
                    # computing the inverse matrix
                    AA_T = A @ A.T
                    B_TB = B.T @ B
                    AA_T_inv = torch.linalg.pinv(AA_T + delta * torch.eye(A.shape[0]).to(A.device)) 
                    B_TB_inv = torch.linalg.pinv(B_TB + delta * torch.eye(A.shape[0]).to(A.device)) 
   
                    grad_A = (1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin 
                    grad_B = (1 / scaling_factor ** 2) * ((torch.eye(B.shape[0]).to(B.device) - B @ B_TB_inv @ B.T) @ grad_B_orin @ AA_T_inv) 
                    equiv_grad = scaling_factor * B @ grad_A + scaling_factor * grad_B @ A
    
                    grad = equiv_grad
                else:
                    grad = p.grad
                    print(n, p.shape, p.grad)
                    

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step_t = state["step"]
                
                step_t += 1
                
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.lerp_(grad.to(torch.bfloat16), 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad.to(torch.bfloat16), grad.conj().to(torch.bfloat16), value=1 - beta2)
                
                step = _get_value(step_t)
                
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = group['lr'] 
                
                bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
                
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(state["max_exp_avg_sq"], exp_avg_sq, out=state["max_exp_avg_sq"])

                    # Use the max. for normalizing running avg. of gradient
                    denom = (state["max_exp_avg_sq"].sqrt() / bias_correction2_sqrt).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                
                if len(param_list) != 2:
                    
                    if group['weight_decay'] != 0:
                        p.mul_(1 - group["weight_decay"] * group["lr"])
                    
                    p.addcdiv_(exp_avg / bias_correction1, denom, value=-step_size)
                else:
                    g = (exp_avg / bias_correction1) / denom
                    g = g.to(B.dtype)
                    grad_A_orin = scaling_factor * B.T @ g
                    grad_B_orin = scaling_factor * g @ A.T
                    
                    AA_T_inv = torch.linalg.pinv(AA_T + delta * torch.eye(A.shape[0]).to(A.device))
                    B_TB_inv = torch.linalg.pinv(B_TB + delta * torch.eye(A.shape[0]).to(A.device))  

                    if group['X_mode'] == "sylvester":
                        X = solve_sylvester(B.T @ B, A @ A.T, -(1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin @ A.T)
                    elif group['X_mode'] == "symmetry":
                        X = -0.5 * (1 / scaling_factor ** 2) * B_TB_inv @ B.T @ grad_B_orin @ AA_T # symmetry
                    else:
                        X = torch.zeros((B_TB_inv.shape[0], B_TB_inv.shape[0])).to(B.device)
                    X = torch.tensor(X).to(B.device).to(B.dtype)
    
                    grad_A = (1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin + X @ A
                    grad_B = (1 / scaling_factor ** 2) * ((torch.eye(B.shape[0]).to(B.device) - B @ B_TB_inv @ B.T) @ grad_B_orin @ AA_T_inv) - B @ X
                
                    if group['weight_decay'] != 0:
                        B.mul_(math.sqrt(1 - group["weight_decay"] * group["lr"]))
                        A.mul_(math.sqrt(1 - group["weight_decay"] * group["lr"]))
                        param_dict[weight_name].mul_(1 - group["weight_decay"] * group["lr"])
                
                    param_list[0].add_(grad_A, alpha=-step_size)
                    param_list[1].add_(grad_B, alpha=-step_size)
                    param_list = []
    def is_same(self, name_list):
        return (name_list[0].split('.')[:-3] == name_list[1].split('.')[:-3])
        
    def step_efficient(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            scaling_factor = group["scaling_factor"]

            param_list = []
            name_list = []
            for p, n in zip(group["params"], group["names"]):
                if p.grad is None:
                    continue

                if 'lora' in n:
                    param_list.append(p)
                    name_list.append(n)
                    if len(param_list) == 2:
                        name = n[: n.find('lora')] + 'lora'
                    elif len(param_list) == 1:
                        continue
                else:
                    name = n
                
                state = self.state[name]
            
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    if len(param_list) == 2:
                        state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                        # Exponential moving average of gradient values
                        state["exp_avg_A"] = torch.zeros(param_list[0].shape).to(p.device).to(p.dtype)
                        state["exp_avg_B"] = torch.zeros(param_list[1].shape).to(p.device).to(p.dtype)

                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq_A"] = torch.zeros(param_list[0].shape).to(p.device).to(p.dtype)
                        state["exp_avg_sq_B"] = torch.zeros(param_list[1].shape).to(p.device).to(p.dtype)

                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq_A"] = torch.zeros(param_list[0].shape).to(p.device).to(p.dtype)
                            state["max_exp_avg_sq_B"] = torch.zeros(param_list[1].shape).to(p.device).to(p.dtype)
                    else:
                        state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros(p.shape).to(p.device).to(p.dtype)

                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros(p.shape).to(p.device).to(p.dtype)

                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros(p.shape).to(p.device).to(p.dtype)

                # step
                if len(param_list) == 2:
                    # list_param = [A, B]
                    A = param_list[0]
                    B = param_list[1]
                    grad_A_orin = A.grad
                    grad_B_orin = B.grad
                    
                    # projection
                    delta = 1e-8
 
                    # computing the inverse matrix
                    AA_T = A @ A.T
                    B_TB = B.T @ B
                    AA_T_inv = torch.linalg.pinv(AA_T + delta * torch.eye(A.shape[0]).to(A.device)) 
                    B_TB_inv = torch.linalg.pinv(B_TB + delta * torch.eye(A.shape[0]).to(A.device))
   
                    if group['X_mode'] == "sylvester":
                        X = solve_sylvester(B.T @ B, A @ A.T, -(1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin @ A.T)
                    elif group['X_mode'] == "symmetry":
                        X = -0.5 * (1 / scaling_factor ** 2) * B_TB_inv @ B.T @ grad_B_orin @ AA_T # symmetry
                    else:
                        X = torch.zeros((B_TB_inv.shape[0], B_TB_inv.shape[0])).to(B.device)
                    X = torch.tensor(X).to(B.device).to(B.dtype)
    
                    grad_A = (1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin + X @ A
                    grad_B = (1 / scaling_factor ** 2) * ((torch.eye(B.shape[0]).to(B.device) - B @ B_TB_inv @ B.T) @ grad_B_orin @ AA_T_inv) - B @ X
                    
                    exp_avg_A = state["exp_avg_A"]
                    exp_avg_sq_A = state["exp_avg_sq_A"]
                    
                    exp_avg_B = state["exp_avg_B"]
                    exp_avg_sq_B = state["exp_avg_sq_B"]

                    step_t = state["step"]

                    step_t += 1
            
                    exp_avg_A.lerp_(grad_A, 1 - beta1)
                    exp_avg_B.lerp_(grad_B, 1 - beta1)
                    exp_avg_sq_A.mul_(beta2).addcmul_(grad_A, grad_A.conj(), value=1 - beta2)
                    exp_avg_sq_B.mul_(beta2).addcmul_(grad_B, grad_B.conj(), value=1 - beta2)

                    step = _get_value(step_t)
                    
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step

                    step_size = group['lr'] 

                    bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
                    
                    if group['amsgrad']:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.maximum(state["max_exp_avg_sq_A"], exp_avg_sq_A, out=state["max_exp_avg_sq_A"])
                        torch.maximum(state["max_exp_avg_sq_B"], exp_avg_sq_B, out=state["max_exp_avg_sq_B"])

                        # Use the max. for normalizing running avg. of gradient
                        denom_A = (state["max_exp_avg_sq_A"].sqrt() / bias_correction2_sqrt).add_(group['eps'])
                        denom_B = (state["max_exp_avg_sq_B"].sqrt() / bias_correction2_sqrt).add_(group['eps'])
                    else:
                        denom_A = (exp_avg_sq_A.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                        denom_B = (exp_avg_sq_B.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                        
                    if group['weight_decay'] != 0:
                        A.mul_(1 - group["weight_decay"] * group["lr"])
                        B.mul_(1 - group["weight_decay"] * group["lr"])
                        
                    A.addcdiv_(exp_avg_A / bias_correction1, denom_A, value=-step_size)
                    B.addcdiv_(exp_avg_B / bias_correction1, denom_B, value=-step_size)
                    param_list = []  
                    name_list = []
                else:
                    grad = p.grad
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    step_t = state["step"]

                    step_t += 1

                    # Decay the first and second moment running average coefficient
                    # In-place operations to update the averages at the same time
                    exp_avg.lerp_(grad, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

                    step = _get_value(step_t)

                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step

                    step_size = group['lr'] 

                    bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
                    if group['weight_decay'] != 0:
                        p.mul_(1 - group["weight_decay"] * group["lr"])
                    
                    p.addcdiv_(exp_avg / bias_correction1, denom, value=-step_size)
                    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.mode == "full":
            self.step_full()
        else:
            self.step_efficient()

        return loss


class SGD(Optimizer):
    
    def __init__(
        self,
        named_params: Iterable[nn.parameter.Parameter],
        scaling_factor: float = 2.,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov=False,
        *,
        maximize: bool = False, 
        differentiable: bool = False,
        X_mode: str = "sylvester",
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")    

        names = []
        params = []
        for n, p in named_params:
            names.append(n)
            params.append(p)
        self.X_mode = X_mode
        
        defaults = dict(
            lr=lr,
            names=names,
            scaling_factor=scaling_factor,
            X_mode=X_mode,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            dampening = group["dampening"]
            scaling_factor = group["scaling_factor"]

            param_list = []
            param_dict = dict(zip(group["names"], group["params"]))

            for n, p in param_dict.items():
                if p.grad is None:
                    continue
                if 'lora' in n:
                    param_list.append(p)
                    if len(param_list) == 2:
                        base_name = n[: n.find('lora')] 
                        name = base_name + 'lora'
                        weight_name = base_name + "base_layer.weight"
                        size = (param_list[1].shape[0], param_list[0].shape[1])
                    elif len(param_list) == 1:
                        continue
                else:
                    name = n
                    size = p.shape
                
                state = self.state[name]

                # step
                if len(param_list) == 2:
                    # list_param = [A, B]
                    A = param_list[0]
                    B = param_list[1]
                    grad_A_orin = A.grad
                    grad_B_orin = B.grad
                    
                    # projection
                    delta = 1e-8
 
                    # computing the inverse matrix
                    AA_T = A @ A.T
                    B_TB = B.T @ B
                    AA_T_inv = torch.linalg.pinv(AA_T + delta * torch.eye(A.shape[0]).to(A.device)) 
                    B_TB_inv = torch.linalg.pinv(B_TB + delta * torch.eye(A.shape[0]).to(A.device)) 
   
                    if group['X_mode'] == "sylvester":
                        X = solve_sylvester(B.T @ B, A @ A.T, -(1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin @ A.T)
                    elif group['X_mode'] == "symmetry":
                        X = -0.5 * (1 / scaling_factor ** 2) * B_TB_inv @ B.T @ grad_B_orin @ AA_T # symmetry
                    else:
                        X = torch.zeros((B_TB_inv.shape[0], B_TB_inv.shape[0])).to(B.device)
                    X = torch.tensor(X).to(B.device).to(B.dtype)

                    grad_A = (1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin + X @ A
                    grad_B = (1 / scaling_factor ** 2) * ((torch.eye(B.shape[0]).to(B.device) - B @ B_TB_inv @ B.T) @ grad_B_orin @ AA_T_inv) - B @ X
                    step_size = group['lr'] 

                    param_list[0].add_(grad_A, alpha=-step_size)
                    param_list[1].add_(grad_B, alpha=-step_size)
                else:
                    step_size = group['lr'] 
                    grad = p.grad
                    p.add_(grad, alpha=-step_size)


                param_list = []

        return loss
    