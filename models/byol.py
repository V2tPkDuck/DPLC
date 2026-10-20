# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


from models.util import cosine_annealing_LR
from network import backbone_dict
from utils.gather_layer import GatherLayer
from utils.grad_scaler import NativeScalerWithGradNormCount



class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent A New Approach to Self-Supervised Learning
    https://github.com/lucidrains/byol-pytorch/tree/master/byol_pytorch
    """

    def __init__(self,opt):
        nn.Module.__init__(self)
        encoder_type, dim_in = backbone_dict[opt.encoder_name]
        encoder = encoder_type()
        self.symmetric = True
        self.shuffling_bn = opt.dist 
        self.m = opt.momentum_base
        self.num_cluster = opt.num_cluster
        self.temperature = opt.temperature
        self.fea_dim = opt.fea_dim

        # create the encoders
        self.encoder_q = encoder
        self.projector_q = nn.Sequential(
            nn.Linear(dim_in, opt.hidden_size),
            nn.BatchNorm1d(opt.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.hidden_size, opt.fea_dim)
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.projector_k = copy.deepcopy(self.projector_q)

        self.predictor = nn.Sequential(nn.Linear(opt.fea_dim, opt.hidden_size),
                                       nn.BatchNorm1d(opt.hidden_size),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(opt.hidden_size, opt.fea_dim)
                                       )
        self.q_params = list(self.encoder_q.parameters()) + list(self.projector_q.parameters())
        self.k_params = list(self.encoder_k.parameters()) + list(self.projector_k.parameters())

        for param_q, param_k in zip(self.q_params, self.k_params):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for m in self.predictor.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                                nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.encoder = nn.Sequential(self.encoder_k, self.projector_k)


        self.encoder_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
        self.projector_q = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_q)
        self.predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)
        self.feature_extractor_copy = copy.deepcopy(self.encoder).cuda()


        self.scaler = NativeScalerWithGradNormCount(amp=opt.amp)




    def forward_k(self, im_k):
        # compute key features
        with torch.no_grad():  # no gradient to keys
            if self.shuffling_bn:
                # shuffle for making use of BN
                im_k_, idx_unshuffle = self._batch_shuffle_ddp(im_k)
                k = self.encoder_k(im_k_)  # keys: NxC
                k = k.float()
                k = self.projector_k(k)
                k = nn.functional.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            else:
                k = self.encoder_k(im_k)  # keys: NxC
                k = self.projector_k(k)
                k = nn.functional.normalize(k, dim=1)

            k = k.detach_()
            all_k = self.concat_all_gather(k)

        return k, all_k


    def forward_loss(self, im_q, im_k):
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = self.projector_q(q)

        k, all_k = self.forward_k(im_k)
        contrastive_loss = (2 - 2 * F.cosine_similarity(self.predictor(q), k)).mean()
        
        # Avoid using GatherLayer in single-process mode (no distributed init)
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            all_q = F.normalize(torch.cat(GatherLayer.apply(q), dim=0), dim=1)
        else:
            all_q = F.normalize(q, dim=1)
        
        return contrastive_loss, all_q, all_k
    def forward(self, im_q, im_k, momentum_update=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if self.symmetric:
            contrastive_loss1, q1, k1 = self.forward_loss(im_q, im_k)
            contrastive_loss2, q2, k2 = self.forward_loss(im_k, im_q)
            contrastive_loss = 0.5 * (contrastive_loss1 + contrastive_loss2)
            # cluster_loss_batch = 0.5 * (cluster_loss_batch1 + cluster_loss_batch2)
            q = torch.cat([q1, q2], dim=0)
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            contrastive_loss, q, k = self.forward_loss(im_q, im_k)

        # if momentum_update:
        #     # update the key encoder
        #     with torch.no_grad():  # no gradient to keys
        #         self._momentum_update_key_encoder()

        return contrastive_loss, q





    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.q_params, self.k_params):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized() \
        or torch.distributed.get_world_size() == 1:
            return x, torch.arange(x.shape[0], device=x.device)   # dummy unshuffle index
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized() \
        or torch.distributed.get_world_size() == 1:
            return x   # dummy unshuffle index
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        Falls back to a no-op when distributed is not initialized or world_size == 1.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized() \
                or torch.distributed.get_world_size() == 1:
            return tensor
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output
def train_byol(opt,model,optimizer,train_loader,epoch,log):
    n_iter = opt.num_batch*(epoch-1)+1
    optimizer.zero_grad(set_to_none=True)  # clear once before the epoch
    for i, (inputs, target, idx) in enumerate(train_loader):
        inputs_1, inputs_2 = inputs
        inputs_1 = inputs_1.cuda(non_blocking=True)
        inputs_2 = inputs_2.cuda(non_blocking=True)
        model.train()
        lr = adjust_learning_rate(opt,model,optimizer,n_iter)
        update_params = (n_iter % opt.acc_grd_step == 0)
        with torch.autocast('cuda', enabled=opt.amp):
            loss, q = model(inputs_1, inputs_2)
        loss = loss / opt.acc_grd_step
        model.scaler(loss, optimizer=optimizer, update_grad=update_params)
        if update_params:
            # If the custom scaler does not call zero_grad internally, do it here
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            # handle DDP wrapper
            byol_module = model.module if hasattr(model, 'module') else model
            byol_module._momentum_update_key_encoder()
        if i == 0:
            log.info('Epoch {} loss: {} lr: {}'.format(epoch, loss,lr))
        n_iter += 1



def adjust_learning_rate(opt, model, optimizer, n_iter):
    lr = cosine_annealing_LR(opt, n_iter)
    if opt.fix_predictor_lr:
        predictor_lr = opt.learning_rate
    else:
        predictor_lr = lr * opt.lambda_predictor_lr
    flag = False
    for param_group in optimizer.param_groups:
        if 'predictor' in param_group['name']:
            # flag = True
            param_group['lr'] = predictor_lr
        else:
            param_group['lr'] = lr
    # assert flag

    ema_momentum = opt.momentum_base
    if opt.momentum_increase:
        ema_momentum = opt.momentum_max - (opt.momentum_max - ema_momentum) * (
                np.cos(np.pi * n_iter / (opt.epochs * opt.num_batch)) + 1) / 2
    model.m = ema_momentum
    return lr
def adjust_learning_rate_self_labeling(opt, model, optimizer, n_iter):
    # lr = cosine_annealing_LR(opt, n_iter)
    lr = opt.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr