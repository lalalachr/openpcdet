import argparse
import glob
from multiprocessing import get_logger
from pathlib import Path
import tqdm
import yaml
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from tools.train_utils.optimization import build_optimizer


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--la', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')

    args = parser.parse_args()
    cfg_from_yaml_file(args.la, cfg) 
    # cfg_file = args.la
    # with open(cfg_file,'r') as f:
    #     load_cfg = yaml.safe_load(f)
    #     cfg.update(load_cfg)

    return args,cfg

def main():
    args,cfgs = parse_config()
    # print(args.la)
    # print(cfgs.DATA_CONFIG)
    dataset_cfg = cfgs.DATA_CONFIG
    #print(dataset_cfg.DATASET)

    train_set, train_loader, train_sampler = build_dataloader(  # train_loader返回训练数据,是一个轮次epoch所包含的批次
        dataset_cfg=cfg.DATA_CONFIG,                # 
        class_names=cfg.CLASS_NAMES,                # 识别对象名字
        batch_size=args.batch_size,                 # 每个训练步中使用的样本数量
        dist=False, workers=1,                      # 是否进行分布式训练    工作进程数
        training=True,                              #  是否是训练模式
        merge_all_iters_to_one_epoch=False,         # 否将所有迭代合并为一个周期
        total_epochs=args.epochs,                             # 总的训练周期数
        seed = None                                 # 随机种子
    )
    # for load in train_loader:                     # 一个load代表一个批次数据，根据batch_size决定
    #     print(load)
    #     break
    print(len(train_loader))                        # 如果batch_size=1 那len就是训练数据数量,
    dataloader_iter = iter(train_loader)             # 转换为一个迭代器,转换为迭代器之后就可以直接用next返回下一个批次，不然要用for train_loader in 来获取
    batch = next(dataloader_iter)                     # 获取下一个批次的数据batch
    # print(batch.items())                             # items就直接是一个键值对，直接batch是一个字典
    # points = batch['points']                          # 一张点云图的点云(59250, 5)
    # print(points.shape)


    # assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
    # train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=args.epochs) # 将所有迭代合并到一个 epoch 中
    # total_it_each_epoch = len(train_loader) // max(args.epochs, 1)
    # print(total_it_each_epoch)

    # with tqdm.trange(0, 10, desc='epochs', dynamic_ncols=True) as tbar:
    #     print(tbar)
    #     # for cur_epoch in tbar:
    #     #     print(cur_epoch)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    # print(model)
    
    model.cuda()
    model.train()
    ret_dict, tb_dict, disp_dict = model.forward(batch)
    print(ret_dict)

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)    # 使用自定义adam梯度下降
    # num = sum([m.numel() for m in model.parameters()])
    # num_elements = [m.numel() for m in model.parameters()]
    # for param in model.parameters():                      # 用这个函数看要训练的参数
    #     #print(param)                                     # param看具体参数，param.shape看参数类型
    # print(num_elements)                                   # 返回的是列表 sum之后就是总数   
    
    # ckpt_dir =  Path('..')/'output'/'cfgs'/'custom_models'/'pointpillar'/'star_sever'/'ckpt' # Path('../output/cfgs/custom_models/pointpillar/star_sever/ckpt')
    # ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))                                           # 要用Path才能拼接，前面这种算拼接，后面算直接定义路径
    # print(ckpt_list[-1])
    # print(len(ckpt_list))

    # logger = get_logger()
    # start_epoch = model.load_params_with_optimizer(ckpt_list[-1], to_cpu=False, optimizer=optimizer, logger = logger)    #ckpt_list[-1]就是最后一个模型.pth是指pytorch)
    # print(start_epoch)

    # for x in cfgs.OPTIMIZATION.DECAY_STEP_LIST:
    #     print(x)

if __name__ == '__main__':
    main()
