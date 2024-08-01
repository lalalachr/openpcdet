import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn') # 使所有GPU使用相同的均值和方差
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)                         # 将cfg文件处理为可以像访问属性一样访问字典的键值对，然后合并所有yaml文件，意思是custom_dataset.yaml会被合并上去
    cfg.TAG = Path(args.cfg_file).stem                             # 文件名 比如就是pointpillar
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml' 
    # 例如划分cfgs/custom_models/pointpillar.yaml，得到custom_models
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:                                 # 如果有额外的配置项，将这些配置项应用到 cfg.yaml 中,就是通过命令行改写.yaml
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none': # 据参数 args.launcher 的值，使用相应的分布式训练方案初始化训练环境，并返回总 GPU 数和本地 GPU 的索引。
        dist_train = False      # 表示不进行分布式训练。
        total_gpus = 1          # 表示仅使用一块 GPU 进行训练py。
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU 
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus' 
        args.batch_size = args.batch_size // total_gpus                                  # // 整除 

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs    # 设置训练轮数

    if args.fix_random_seed:                                                             # 设置随机种子的
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    # 结果保存路径    
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag # args.extra_tag是自定义的保存目录
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))


    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL' # 获取状态变量？
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')
        
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")     # 创建数据加载器，网络，加速器
    train_set, train_loader, train_sampler = build_dataloader(                         # train_loader（一个DataLoader对象）返回批次的数据 for load in train_loader，一个load代表一个批次，
        dataset_cfg=cfg.DATA_CONFIG,                # 数据配置                          # train_set是数据集，CustomDataset对象
        class_names=cfg.CLASS_NAMES,                # 识别对象名字                      # train_sampler 加载数据的机制
        batch_size=args.batch_size,                 # 每个训练步中使用的样本数量         # train_loader是一个epoch所包含的批次
        dist=dist_train, workers=args.workers,      # 是否进行分布式训练    工作进程数
        logger=logger,
        training=True,                              #  是否是训练模式
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, # 否将所有迭代合并为一个周期
        total_epochs=args.epochs,                   # 总的训练周期数
        seed=666 if args.fix_random_seed else None  # 随机种子
    )
    # 加载网络模型 model是PointPillar对象
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)

    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # 检查是否启用了同步批量归一化，同步归一化即使所有GPU使用相同的均值和方差，默认关闭
    model.cuda()

    # 加载梯度优化器 返回OptimWrapper对象 采用adam_onecycle 一种改进的 Adam 优化器策略
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)            # 使用自定义adam梯度下降

    # load checkpoint if it is possible 检查要在某个已有的模型上训练
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:                           # 加载预训练模型的参数到当前模型中
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:                                       # 用于加载检查点（checkpoint）以恢复训练
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1                                # 检查是否提供了一个特定的检查点路径（args.ckpt）。如果提供了，则直接加载该检查点
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))              # 如果没有提供检查点路径，则会搜索指定目录（ckpt_dir）中的所有检查点文件，并选择最后修改时间最晚的文件作为检查点
              
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)                    # 按时间排序
            while len(ckpt_list) > 0:
                try:    
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger    #ckpt_list[-1]就是最后一个模型.pth是指pytorch
                    )
                    last_epoch = start_epoch + 1
                    break                                           # 没有异常直接跳出循环了
                except:                                             # 发生异常才会执行
                    ckpt_list = ckpt_list[:-1]                      # 移除最后一个元素
                    last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters 将模型设置为训练模式
    if dist_train: # 分布式训练
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------') # model.parameters()获取可训练的参数的个数，这里是4824108个
    logger.info(model)

    # 一个 OneCycle 学习率调度器，根据训练中动态调整学习率和参数 lr_scheduler是OneCycle对象 lr_warmup_scheduler是预热
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,      # total_iters_each_epoch为训练批次，total_epochs是训练轮数
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

      # -----------------------start training---------------------------        
    logger.info('**********************Start training %s/%s(%s)**********************'      # 开始训练
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    train_model(    # 模型训练
        model,                                                                  # 模型
        optimizer,                                                              # 优化器
        train_loader,                                                           # 按批次分的数据
        model_func=model_fn_decorator(),                                        # 训练一次的损失与更新参数
        lr_scheduler=lr_scheduler,                                              # 学习率调度器
        optim_cfg=cfg.OPTIMIZATION,                                             # 优化器参数
        start_epoch=start_epoch,                                                # 开始轮数
        total_epochs=args.epochs,                                               # 总轮数       
        start_iter=it,
        rank=cfg.LOCAL_RANK,                                                    # 这里为0，代表进程的编号
        tb_log=tb_log,                                                          # tensorboard
        ckpt_save_dir=ckpt_dir,                                                 # 保存地址
        train_sampler=train_sampler,                                            # 加载数据的机制
        lr_warmup_scheduler=lr_warmup_scheduler,                                # 预热
        ckpt_save_interval=args.ckpt_save_interval,                             # 每隔多少个epoch或训练步骤保存一次模型的状态
        max_ckpt_save_num=args.max_ckpt_save_num,                               # 最大保存数
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,         # 是否合并为一个周期
        logger=logger,                                                          # log
        logger_iter_interval=args.logger_iter_interval,                         # 日志记录时间间隔 默认50,50个批次记录一次
        ckpt_save_time_interval=args.ckpt_save_time_interval,                   # 保存一次模型的时间间隔，默认300s
        use_logger_to_record=not args.use_tqdm_to_record,                       # 是否使用tqdm，也就是进度条
        show_gpu_stat=not args.wo_gpu_stat,                                     # GPU统计信息
        use_amp=args.use_amp,                                                   # 混合精度训练
        cfg=cfg
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()     # 这段代码的作用是清理共享内存

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %      # 开始评估
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'                   # 评估对应的文件夹
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs计算的结果小于零，则将其设置为零
    
    # 重复评估模型的指定检查点
    repeat_eval_ckpt(  
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    ) 
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
