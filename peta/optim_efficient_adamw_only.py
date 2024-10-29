import math
import torch

from typing import Tuple, Union, List

from torch.optim import Optimizer
from torch._utils import is_compiling
from scipy.linalg import solve_sylvester


def find_lora_names(n):
    for substring in ['lora_A', 'lora_B']:
        if substring in n:
            return substring
    return ""

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

class LoRAProAdamW(Optimizer):
    def __init__(
        self,
        named_params: List[Tuple[str, torch.Tensor]],
        lr: Union[float, torch.Tensor] = 1e-3,
        scaling_factor: float = 2.,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
        X_mode: str = "sylvester",
    ):
        
        """
        Example of named params:
        [{'params':named_param_group1, 'lr':lr1},
        {'params':named_param_group2, 'lr':lr2}]
        """
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
        if not X_mode in ["zero", "sylvester", "symmetry"]:
            raise ValueError(f"Invalid mode value: {X_mode}, mode should be in ['zero', 'sylvester', 'symmetry']")

        self.X_mode = X_mode
        self.step_ = 0
        
        if not isinstance(named_params, list):
            named_params = [named_params]
        # Process named_params into param groups
        params = []

        for named_params_group in named_params:
            param_group = {
                'params': [],
                'names': [],
                'lr': named_params_group.get('lr', lr),
                'scaling_factor': scaling_factor,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay,
                'amsgrad': amsgrad,
                'maximize': maximize,
                'differentiable': differentiable,
                'X_mode': X_mode
            }
            for name, param in named_params_group['params']:
                param_group['params'].append(param)
                param_group['names'].append(name)
                
            params.append(param_group)
        
        defaults = dict(
            lr=lr,
            scaling_factor=scaling_factor,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            differentiable=differentiable,
            X_mode=X_mode,
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
            beta1, beta2 = group["betas"]
            scaling_factor = group["scaling_factor"]

            param_dict = {}
            # param_dict process a group of lora parameter at one time
            for p, n in zip(group["params"], group["names"]):
                if p.grad is None:
                    continue

                lora_weight_name = find_lora_names(n)
                if lora_weight_name:
                    param_dict[lora_weight_name] = p
                    if len(param_dict.keys()) == 2:
                        # weight_a and weight_b share the same state
                        name = n[: n.find(lora_weight_name)] + 'lora'
                    elif len(param_dict.keys()) == 1:
                        continue
                else:
                    name = n
                
                state = self.state[name]
            
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    # All lora states store in one state dict.
                    if len(param_dict.keys()) == 2:
                        state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())

                        # Exponential moving average of gradient values
                        state["exp_avg_A"] = torch.zeros(param_dict['lora_A'].shape).to(p.device).to(p.dtype)
                        state["exp_avg_B"] = torch.zeros(param_dict['lora_B'].shape).to(p.device).to(p.dtype)

                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq_A"] = torch.zeros(param_dict['lora_A'].shape).to(p.device).to(p.dtype)
                        state["exp_avg_sq_B"] = torch.zeros(param_dict['lora_B'].shape).to(p.device).to(p.dtype)

                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq_A"] = torch.zeros(param_dict['lora_A'].shape).to(p.device).to(p.dtype)
                            state["max_exp_avg_sq_B"] = torch.zeros(param_dict['lora_B'].shape).to(p.device).to(p.dtype)
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
                if len(param_dict.keys()) == 2:
                    # list_param = [A, B]
                    A = param_dict['weight_a']
                    B = param_dict['weight_b']
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
    
                    # calculate optimized gradient of A
                    grad_A = (1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin + X @ A
                    grad_B = (1 / scaling_factor ** 2) * ((torch.eye(B.shape[0]).to(B.device) - B @ B_TB_inv @ B.T) @ grad_B_orin @ AA_T_inv) - B @ X
                    
                    # normal adamw computation process
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
                    param_dict['weight_b'] = {}
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

        return loss