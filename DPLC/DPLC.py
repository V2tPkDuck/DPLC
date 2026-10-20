# -*- coding: UTF-8 -*-
import argparse
import os
import time
import torch
import torch.distributed as dist
import numpy as np
from sklearn.decomposition import PCA
import random
from torch import nn
from get_config import get_config
from models.Accuracy import cluster_accuracy, clustering
from models.byol import BYOL
from models.util import get_dataset, logger, get_embedding_for_test
from models.other_backbones import load_ssl_model
from core import (
    Cluster_projector, crp_gibbs, train_cluster_projector_instance_cluster,
    evaluate, save_checkpoint, _EXP_LOG, _sample_gpu_memory, _reset_gpu_memory_stats,
    compute_all_train_predictions, compute_dp_subcluster_weights,
    save_experiment_log,  
)
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture

def worker_init_fn(worker_id):
    """Ensure each DataLoader worker has a reproducible RNG seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(opt):
    if opt.dist:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(dist.get_rank())
        rank = dist.get_rank()
    else:
        rank = 0
    if opt.num_devices > 0:
        assert opt.num_devices == torch.cuda.device_count()  # total batch size
    if os.path.exists(opt.save_dir) is not True:
        os.makedirs(opt.save_dir, exist_ok=True)
    if os.path.exists(opt.cached_feature) is not True:
        os.makedirs(opt.cached_feature, exist_ok=True)
    
    seed = opt.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    rank = torch.distributed.get_rank()
    t_total_start = time.perf_counter()
    _reset_gpu_memory_stats()
    logName = f"log_{opt.backbone}.txt"
    log = logger(path=opt.save_dir, local_rank=rank, log_name=logName)
    log.info(str(opt))
    # if opt.wandb:
    #     wandb.init(project=opt.project_name,entity=opt.entity,name=opt.run_name,config=str(opt))
    # opt.learning_rate = opt.learning_rate * (opt.batch_size / 256)
    
    model, opt.fea_dim = load_ssl_model(opt, log)    
    model.cuda()
    train_datasets = get_dataset(opt, 'mem')
    test_datasets = get_dataset(opt, 'test')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_datasets, shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        pin_memory=True, 
        worker_init_fn=worker_init_fn,)
    opt.num_batch = len(train_loader)
    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        num_workers=opt.num_workers,
        batch_size=opt.batch_size,
        sampler=test_sampler,
        shuffle=False,
        pin_memory=True, 
        worker_init_fn=worker_init_fn,)
    start_epoch = 1
    t_cache_start = time.perf_counter()
    train_labels_path = os.path.join(opt.cached_feature, 'train_labels.npy')
    _EXP_LOG['opt'] = {'dataset': opt.dataset, 'imb_ratio': opt.imb_ratio,
                       'backbone': opt.backbone, 'crp': opt.crp,
                       'alpha': opt.alpha, 'beta': opt.beta,
                       'threshold': opt.threshold, 'batch_size': opt.batch_size,
                       'cluster_num': opt.cluster_num}
    cache_path = os.path.join(opt.cached_feature, 'frozen_features.pth')
    use_cache = (opt.backbone != 'byol')
    if use_cache and os.path.exists(cache_path) and os.path.exists(train_labels_path):
        log.info("Loading cached features from disk.")
        cache = torch.load(cache_path)
        cached_train_feat = cache['train_feat'].cpu()
        cached_test_feat = cache['test_feat'].cpu()
        cached_test_labels = cache['test_labels'].cpu()
        cached_train_labels = torch.from_numpy(np.load(train_labels_path))
    else:
        model.eval()
        with torch.inference_mode():
            cached_train_feat, cached_train_labels = get_embedding_for_test(
                model, train_loader, loader='mem')
            cached_test_feat, cached_test_labels = get_embedding_for_test(
                model, test_loader)
        if use_cache:
            torch.save({'train_feat': cached_train_feat.cpu(),
                        'test_feat': cached_test_feat.cpu(),
                        'test_labels': cached_test_labels.cpu(),
                        'train_labels': cached_train_labels.cpu()}, cache_path)
            np.save(train_labels_path, cached_train_labels.cpu().numpy()) 
            log.info("Feature caching done.")

    t_cache = time.perf_counter() - t_cache_start
    _EXP_LOG['stage_times']['feature_caching'] = t_cache
    _sample_gpu_memory(0, 'after_feature_caching')

    assert opt.train_from_start != opt.train_from_resume
    if opt.train_from_start:
        # log.info("===> Pre-caching static embeddings to skip backbone during clustering/eval...")
        model.eval()
        with torch.no_grad():
            embedding, target = get_embedding_for_test(model, test_loader)
        actual_fea_dim = embedding.shape[1]
        if opt.fea_dim != actual_fea_dim:
            log.info(f"Correcting opt.fea_dim from {opt.fea_dim} to {actual_fea_dim}")
            opt.fea_dim = actual_fea_dim
        model.train()
        np.random.seed(seed)
        y_pred, image_centers = clustering(embedding, opt.cluster_num)
        acc, nmi, ari, ca = cluster_accuracy(target.cpu().numpy(), y_pred.cpu().numpy())
        # print([acc, nmi, ari, np.mean(ca), ca])
        image_centers = torch.nn.functional.normalize(image_centers, dim=-1)

        cluster_projector = Cluster_projector(opt.fea_dim, opt.cluster_num, image_centers)
        cluster_projector.cuda()
        if opt.backbone == 'byol':
            for param in model.parameters():
                param.requires_grad = False
            for param in model.encoder_k.parameters():
                param.requires_grad = True
            for param in model.projector_k.parameters():
                param.requires_grad = True
            params = (list(model.encoder_k.parameters()) +
                    list(cluster_projector.parameters()) +
                    list(model.projector_k.parameters()))
        else:
            # For other backbones, keep the frozen backbone and only train cluster head
            for p in model.get_params():
                p.requires_grad = False
            params = list(cluster_projector.parameters())
        optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        if opt.backbone == 'byol':
            print("Number of param groups:", len(optimizer.param_groups))
            for i, group in enumerate(optimizer.param_groups):
                print(f"Group {i}: {len(group['params'])} tensors")
    elif opt.train_from_resume:
        checkpoint = torch.load(os.path.join(opt.save_dir, opt.checkpoint), map_location="cuda")
        model.load_state_dict(checkpoint['model_state_dict'])
        cluster_projector = Cluster_projector(opt.fea_dim, opt.cluster_num, torch.randn(opt.cluster_num, opt.fea_dim))
        cluster_projector.cuda()
        cluster_projector.load_state_dict(checkpoint['projector_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if opt.backbone == 'byol':
            # For BYOL, unfreeze encoder_k and projector_k to fine-tune them
            # (matching old behavior: using target branch + cluster head)
            for name, param in model.named_parameters():
                if 'encoder' in name or 'projector' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            params = (list(model.encoder_k.parameters()) +
              list(cluster_projector.parameters()) +
              list(model.projector_k.parameters()))        
        else:
            # For other backbones, keep the frozen backbone and only train cluster head
            for p in model.get_params():
                p.requires_grad = False
            params = list(cluster_projector.parameters())
        optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        optimizer.load_state_dict(checkpoint['optim'])
    else:
        raise ValueError('no checkpoint of the encoder or clustering heads')

    # NEW: global DP state
    global_mini_hard = None
    global_soft = None
    results = []
    t_head_training_start = time.perf_counter()
    for epoch in range(start_epoch, opt.epochs + 1):
        train_sampler.set_epoch(epoch)

        # === PERIODIC CRP-DP UPDATE (true Dirichlet Process) ===
        if epoch % 15 == 0 or epoch == 1:
            if opt.backbone == 'byol':
                # Recompute embeddings from model for CRP
                model.eval()
                with torch.no_grad():
                    embedding, _ = get_embedding_for_test(model, test_loader)
                embedding_np = embedding.cpu().numpy()
            else:
                embedding_np = cached_train_feat.cpu().numpy()
            if opt.crp == 'crp':
                global_mini_hard, global_soft, n_occ, crp_time = crp_gibbs(
                    embedding_np,
                    n_sweeps=60,
                    alpha=opt.alpha,
                    num_restarts=3
                )
                _EXP_LOG['crp_epochs'].append(epoch)
                _EXP_LOG['crp_n_occ'].append(n_occ)
                _EXP_LOG['crp_times_sec'].append(crp_time)
                _sample_gpu_memory(epoch, f'after_CRP_epoch{epoch}')
                log.info(f"[Epoch {epoch}] CRP-DP occupied clusters (Nocc): {n_occ}")
            else:   # kmeans
                from sklearn.cluster import KMeans
                n_mini = opt.other_cluster_num
                kmeans = KMeans(n_clusters=n_mini, random_state=opt.seed, n_init=10)
                hard = kmeans.fit_predict(embedding_np)
                soft = np.eye(n_mini)[hard].astype(np.float32)# one-hot
                global_mini_hard, global_soft = hard, soft
                print(f"[K-means] Using {n_mini} sub clusters")

        train_cluster_projector_instance_cluster(
            opt, model, optimizer, train_loader, epoch, log,
            cluster_projector, global_mini_hard, global_soft
        )

        if epoch % 25 == 0 or epoch == 1:
            if opt.backbone == 'byol':
                # Recompute test features with current model
                model.eval()
                with torch.no_grad():
                    cached_test_feat, cached_test_labels = get_embedding_for_test(model, test_loader)
            acc, nmi, ari, ca_mean, per_class_ca, test_preds, test_confs = evaluate(cluster_projector, 
                                    cached_test_feat.to('cuda'), 
                                    cached_test_labels,
                                    log)
            results.append((epoch, acc, nmi, ari, ca_mean))
            _EXP_LOG['eval_results'].append([epoch, acc, nmi, ari, ca_mean])
            _EXP_LOG['per_class_acc_final'] = per_class_ca
            _sample_gpu_memory(epoch, f'after_eval_epoch{epoch}')
            model.train()
            cluster_projector.train()

        if rank == 0:
            if epoch % opt.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'projector_state_dict': cluster_projector.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(opt.save_dir, 'DPLC_{}.pt'.format(epoch)))

    t_head_training = time.perf_counter() - t_head_training_start   # 需要在循环前定义!
    _EXP_LOG['stage_times']['head_training'] = t_head_training
    if rank == 0:
        log.info("Computing final predictions on training set...")
        train_preds, train_confs, _ = compute_all_train_predictions(
            cluster_projector, cached_train_feat
        )
        _EXP_LOG['head_final']['predictions'] = train_preds
        _EXP_LOG['head_final']['confidences'] = train_confs
        if global_mini_hard is not None:
            sizes, weights = compute_dp_subcluster_weights(
                global_mini_hard, alpha=opt.alpha, K_fallback=opt.cluster_num
            )
            _EXP_LOG['dp_final']['hard_assignments'] = global_mini_hard
            _EXP_LOG['dp_final']['n_occ'] = len(sizes)
            _EXP_LOG['dp_final']['subcluster_sizes'] = sizes
            _EXP_LOG['dp_final']['subcluster_weights'] = weights
    t_total = time.perf_counter() - t_total_start
    _EXP_LOG['stage_times']['total'] = t_total
    log.info(f"Total wall time: {t_total:.1f}s ({t_total/3600:.2f}h)")

    if rank == 0 and results:
        save_experiment_log(os.path.join(opt.save_dir, 'experiment_log.json'))
        import shutil
        shutil.copy(
            os.path.join(opt.save_dir, 'experiment_log.pkl'),
            "/home/cuiletian/Github/DPLC1/visualization/cifar10_imb10/experiment_log.pkl"
        )
        shutil.copy(
            os.path.join(opt.save_dir, 'experiment_log.json'),
            "/home/cuiletian/Github/DPLC1/visualization/cifar10_imb10/experiment_log.json"
        )
        log.info("\n" + "="*60)
        try:
            from visualize_all import visualize_all
            viz_dir = "/mnt/d/Github/DPLC/visualization/cifar10_imb10"
            os.makedirs(viz_dir, exist_ok=True)
            visualize_all(
                log_pkl_path=os.path.join(opt.save_dir, 'experiment_log.pkl'),
                output_dir=viz_dir,
                cached_train_feat_path=cache_path,
                train_labels_path=train_labels_path,
            )
            log.info(f"Visualizations saved to {viz_dir}")
        except Exception as e:
            log.warning(f"Viz failed: {e}")
        log.info("Evaluation Results Table (every 25 epochs, including epoch 1)")
        log.info("Epoch\tACC\t\tCAA\t\tNMI\t\tARI")
        log.info("-"*60)
        for idx, (epoch, acc, nmi, ari, ca_mean) in enumerate(results):
            log.info(f"{epoch}\t\t\033[92m{acc:.4f}\033[0m\t\t\033[92m{ca_mean:.4f}\033[0m\t\t\033[92m{nmi:.4f}\033[0m\t\t\033[92m{ari:.4f}\033[0m")
        log.info("="*60)
        log.info("\n" + "="*60)
        log.info("Hyperparameters")
        log.info("-"*60)
        log.info(f"Dataset              : {opt.dataset}")
        log.info(f"CRP or K-means       : {opt.crp}")
        log.info(f"Imbalance ratio      : {opt.imb_ratio}")
        log.info(f"Learning rate        : {opt.learning_rate}")
        log.info(f"Batch size           : {opt.batch_size}")
        log.info(f"Threshold            : {opt.threshold}")
        log.info(f"Alpha (α)            : {opt.alpha}")
        log.info(f"Beta (β)             : {opt.beta}")
        log.info(f"Minimum ratio (δ)    : {opt.minimum_ratio}")
        log.info(f"Mini-cluster number M: {opt.other_cluster_num}")
        log.info("="*60)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    global opt
    opt = get_config()
    main(opt)