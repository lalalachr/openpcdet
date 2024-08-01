from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle, CosineAnnealing


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':       # Adam 优化器 lr学习率 WEIGHT_DECAY权重衰减
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':      # SGD 优化器
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER in ['adam_onecycle','adam_cosineanneal']:
        def children(m: nn.Module):         # 自定义优化器，结合了 OneCycle 或 CosineAnnealing 学习率调度策略 定义了一些辅助函数来处理模型层，并使用 OptimWrapper 包装优化器
            return list(m.children())       # 这个函数返回一个模块的子模块列表

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]     #lambda匿名函数(一次性函数) map展开(递归展开) 这两个都还是函数，为了把模型的层展开？
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]        # 用于将展平后的模型子模块封装为一个 nn.Sequential 对象，并将其作为单独的组返回。
        # a = get_layer_groups(model)
        # print(a)
        betas = optim_cfg.get('BETAS', (0.9, 0.99))                            # 如果'BETAS'存在，取配置文件的值，如果没有就取(0.9, 0.99)
        betas = tuple(betas)                                                   # 将 betas 转换为元组，确保 betas 是一个不可变的序列
        optimizer_func = partial(optim.Adam, betas=betas)                      # 建一个部分应用的优化器函数。betas分别控制一阶和二阶动量项的衰减率
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )                                                                       # true_wd权重衰减，防止模型过拟合
    else:
        raise NotImplementedError

    return optimizer

# 构建学习率调度器，在训练过程中动态调整学习率
def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]   # 计算学习率衰减的批次(对应的轮数*总批次)
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:                           # 到达对应的批次后，学习率会乘以LR_DECAY，返回的值就是乘以的比例
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)  # 限幅

    lr_warmup_scheduler = None                                   # 预热
    total_steps = total_iters_each_epoch * total_epochs          # 总批次
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(                                # LR学习率，MOMS动量幅值，DIV_FACTOR
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    elif optim_cfg.OPTIMIZER == 'adam_cosineanneal':
        lr_scheduler = CosineAnnealing(
            optimizer, total_steps, total_epochs, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.PCT_START, optim_cfg.WARMUP_ITER
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
