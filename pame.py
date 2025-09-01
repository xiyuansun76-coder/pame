import math
import torch
from torch.optim import Optimizer


class PAME(Optimizer):
    def __init__(self, params, lr=1e-3, base_beta1=0.9, base_beta2=0.999,
                 gamma=0.99, eps=1e-8, weight_decay=0, max_grad_norm=0):
        # 参数验证
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= base_beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {base_beta1}")
        if not 0.0 <= base_beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {base_beta2}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma parameter: {gamma}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        if not 0.0 <= max_grad_norm:
            raise ValueError(f"Invalid max grad norm: {max_grad_norm}")

        defaults = dict(lr=lr, base_beta1=base_beta1, base_beta2=base_beta2,
                        gamma=gamma, eps=eps, weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)
        super(PAME, self).__init__(params, defaults)

        self.global_step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            # 全局梯度裁剪
            if group['max_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(group['params'], group['max_grad_norm'])

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('PAME does not support sparse gradients')

                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                state['step'] += 1

                # 解耦权重衰减 (AdamW风格)
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # 计算动态beta
                t = state['step']
                beta1 = group['base_beta1'] * (1 - group['gamma'] ** t)
                beta2 = group['base_beta2'] * (1 - group['gamma'] ** t)

                # 平滑过渡替代硬截断
                beta1 = min(beta1, 0.999)
                beta2 = min(beta2, 0.9999)

                # 更新一阶矩
                m = state['m']
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新二阶矩 (添加eps增强数值稳定性)
                v = state['v']
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2).add_(group['eps'])

                # 偏差修正
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                # 计算自适应学习率
                lr_t = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # 更新参数
                denom = v.sqrt()
                p.addcdiv_(m, denom, value=-lr_t)

        return loss

    def get_current_betas(self):
        """获取当前全局步数的动态β值"""
        if self.global_step == 0:
            return self.defaults['base_beta1'], self.defaults['base_beta2']

        t = self.global_step
        gamma = self.defaults['gamma']
        beta1 = self.defaults['base_beta1'] * (1 - gamma ** t)
        beta2 = self.defaults['base_beta2'] * (1 - gamma ** t)
        return min(beta1, 0.999), min(beta2, 0.9999)
