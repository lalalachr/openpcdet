import os

import torch
import tqdm
import time
import glob
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, 
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None, 
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False, use_amp=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch   # accumulated_iter累计训练迭代次数的变量，
    #假设 accumulated_iter为1050，total_it_each_epoch为100，则：计算结果为50。这意味着当前迭代是第11个周期（epoch）中的第50次迭代

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))   # 用于启用或禁用自动混合精度（AMP），use_amp=False代表禁用
    
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True) # leave代表是否保留进度条，每轮最后一个批次时leave_pbar=true
        data_time = common_utils.AverageMeter()         # 用于记录和计算数据加载时间的平均值
        batch_time = common_utils.AverageMeter()        # 用于记录和计算每个批次处理时间的平均值
        forward_time = common_utils.AverageMeter()      # 用于记录和计算前向传播时间的平均值
        losses_m = common_utils.AverageMeter()          # 用于记录和计算损失值的平均值
        one_epoch_loss = 0                                                     # LALALA 2024/7/18添加，查看每一轮的损失值
    
    end = time.time()
    for cur_it in range(start_it, total_it_each_epoch): # for 开始训练一个批次
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)        # 如果这个批次数据用尽，则获取下个批次数据
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter, cur_epoch)  # 根据优化器算法调整学习率

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()   # 将优化器中所有参数的梯度缓存置零

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss, tb_dict, disp_dict = model_func(model, batch)         # 如果use_amp=false 会按照默认浮点精度执行，如果=true，会按照双精度执行
        
        scaler.scale(loss).backward()                                   # scaler 是一个 torch.cuda.amp.GradScaler 对象，这了scale方法如果use_amp=flase，就会直接返回loss，所以会执行loss.backward()
        scaler.unscale_(optimizer)                                      # 如果use_amp=false 直接return
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)   # 防止梯度爆炸
        scaler.step(optimizer)                                          # 返回optimizer.step
        scaler.update()                                                 # 如果use_amp=false 直接return
        
        accumulated_iter += 1
 
        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            batch_size = batch.get('batch_size', None)
            
            data_time.update(avg_data_time)             # 更新时间
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            losses_m.update(loss.item() , batch_size)   # 更新损失值的平均值
            
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })
            
            if use_logger_to_record:
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)
                    
                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '                 # val当前值，avg平均值
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} ' 
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch+1,total_epochs, 100. * (cur_epoch+1) / total_epochs,
                            cur_it,total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m,
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            batch_time=batch_time
                            )
                    )
                    
                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:                    # 显示GPU状态信息
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:                
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()
            
            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
                one_epoch_loss = one_epoch_loss + loss                                                      # LALALA 2024/7/18添加，查看每一轮的损失值
                tb_log.add_scalar('train/one_epoch_loss', one_epoch_loss, cur_epoch)

            # save intermediate ckpt every {ckpt_save_time_interval} seconds                # 根据设定时间间隔保存模型
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1
          
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False, use_amp=False,
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None, show_gpu_stat=False, cfg=None):
    accumulated_iter = start_iter

    # use for disable data augmentation hook
    hook_config = cfg.get('HOOK', None)             # 数据增强相关
    augment_disable_flag = False

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:  # 进度条 desc是进度条描述
        total_it_each_epoch = len(train_loader)     # 一个轮次所包含的批次
        if merge_all_iters_to_one_epoch:            # 这里如果merge_all_iters_to_one_epoch是True的话，在上面的build_dataloader就已经被合并了，所以这里的train_loader就已经是所有轮次的总批次了
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs) # 将所有迭代合并到一个 epoch 中  
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)                    # 这里还是为了表示为一个轮次的批次

        dataloader_iter = iter(train_loader)        # 迭代器
        for cur_epoch in tbar:                      # for 开始训练一个轮次
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:         # 预热
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            
            augment_disable_flag = disable_augmentation_hook(hook_config, dataloader_iter, total_epochs, cur_epoch, cfg, augment_disable_flag, logger)  # 数据增强相关？
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),         # 布尔表达式，如果表达式为真，则leave_pbar为True，检测是不是最后一个批次
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, 
                
                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record, 
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval, 
                show_gpu_stat=show_gpu_stat,
                use_amp=use_amp
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)


def disable_augmentation_hook(hook_config, dataloader, total_epochs, cur_epoch, cfg, flag, logger):
    """
    This hook turns off the data augmentation during training.
    """
    if hook_config is not None:
        DisableAugmentationHook = hook_config.get('DisableAugmentationHook', None)
        if DisableAugmentationHook is not None:
            num_last_epochs = DisableAugmentationHook.NUM_LAST_EPOCHS
            if (total_epochs - num_last_epochs) <= cur_epoch and not flag:
                DISABLE_AUG_LIST = DisableAugmentationHook.DISABLE_AUG_LIST
                dataset_cfg=cfg.DATA_CONFIG
                logger.info(f'Disable augmentations: {DISABLE_AUG_LIST}')
                dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = DISABLE_AUG_LIST
                dataloader._dataset.data_augmentor.disable_augmentation(dataset_cfg.DATA_AUGMENTOR)
                flag = True
    return flag