import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcdet.utils import common_utils

from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
# from .nuscenes.nuscenes_dataset import NuScenesDataset
# from .waymo.waymo_dataset import WaymoDataset
# from .pandaset.pandaset_dataset import PandasetDataset
# from .lyft.lyft_dataset import LyftDataset
# from .once.once_dataset import ONCEDataset
# from .argo2.argo2_dataset import Argo2Dataset
from .custom.custom_dataset import CustomDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
#    'NuScenesDataset': NuScenesDataset,
#    'WaymoDataset': WaymoDataset,
#    'PandasetDataset': PandasetDataset,
#    'LyftDataset': LyftDataset,
#    'ONCEDataset': ONCEDataset,
    'CustomDataset': CustomDataset,
#    'Argo2Dataset': Argo2Dataset
}


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

# 加载数据
def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    # 查是否在分布式环境中运行。如果是，则进入分布式采样器的设置
    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)      # sampler一种用于控制数据加载顺序和分配的机制 
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)  # 不打乱
    else:
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,   #  sampler=none 且training=True 时 打乱
        drop_last=False, sampler=sampler, timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    return dataset, dataloader, sampler
