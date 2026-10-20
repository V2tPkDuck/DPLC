# -*- coding: UTF-8 -*-
import os
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from PIL import ImageFile

from models.byol import BYOL, train_byol
from models.util import get_dataset, logger, collect_params
from get_config import get_config

ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_checkpoint(state, filename='weight.pt'):
    """Save training checkpoint."""
    torch.save(state, filename)


def setup_distributed(opt):
    """Initialize distributed training if enabled.

    If `opt.dist` is True but the required environment variables
    (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) are not set,
    we automatically fall back to single-process mode.
    """
    if opt.dist:
        required_env = ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
        if all(var in os.environ for var in required_env):
            dist.init_process_group(backend='nccl', init_method='env://')
            torch.cuda.set_device(dist.get_rank())
            opt.num_devices = torch.cuda.device_count()
        else:
            print("Distributed training requested but env vars not found. "
                  "Falling back to single GPU mode.")
            opt.dist = False
            opt.num_devices = 1
    else:
        opt.num_devices = 1


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def worker_init_fn(worker_id):
    """Ensure each DataLoader worker has a reproducible RNG seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    # Get unified configuration
    opt = get_config(byol=True)
    opt.epochs = 1000
    # Prepare save directory and logging
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)
    
    setup_distributed(opt)
    set_seed(opt.seed)
    
    rank = dist.get_rank() if opt.dist else 0
    log = logger(path=opt.save_dir, local_rank=rank, log_name="log_imb10.txt")
    log.info(str(opt))
    
    # Model, optimizer, and data loader
    raw_model = BYOL(opt).cuda()
    if opt.dist:
        model = torch.nn.parallel.DistributedDataParallel(
            raw_model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=True,   # key branch parameters do not get gradients
        )
    else:
        model = raw_model
        
    optimizer = torch.optim.SGD(
        params=collect_params(model, exclude_bias_and_bn=opt.exclude_bias_and_bn),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay
    )
    
    train_dataset = get_dataset(opt, 'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if opt.dist else None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    opt.num_batch = len(train_loader)
    
    # Resume training if requested
    start_epoch = 1
    if opt.resume:
        checkpoint_path = opt.checkpoint if opt.checkpoint else os.path.join(opt.save_dir, 'model.pt')
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(opt.save_dir, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])
            log.info(f"Resumed checkpoint {opt.checkpoint} from epoch {checkpoint['epoch']}")
        else:
            log.warning("Checkpoint missing epoch/optim, starting from scratch")
    
    # Training loop
    for epoch in range(start_epoch, opt.epochs + 1):
        if opt.dist:
            train_sampler.set_epoch(epoch)
        train_byol(opt, model, optimizer, train_loader, epoch, log)
        
        if rank == 0:
            # Save checkpoint periodically
            if epoch % 2 == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(opt.save_dir, 'model.pt'))
            
            if epoch % opt.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(opt.save_dir, f'model_{epoch}.pt'))


if __name__ == '__main__':
    if 'LOCAL_RANK' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()