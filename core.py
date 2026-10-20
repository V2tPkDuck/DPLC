import argparse
import os
import time
from tqdm import tqdm
import json
import torch
import torch.distributed as dist
import numpy as np
from numba import njit
from sklearn.decomposition import PCA
import random
from torch import nn
from models.Accuracy import cluster_accuracy, clustering
from models.byol import BYOL
from models.util import get_dataset, logger, get_embedding_for_test
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture

_EXP_LOG = {
    'crp_epochs': [],          # epoch numbers when CRP runs
    'crp_n_occ': [],           # number of occupied sub-clusters
    'crp_times_sec': [],       # wall time per CRP call
    'batch_times': {           # per-batch timing (avg per epoch)
        'fwd': [],             # model forward (both views)
        'head': [],            # cluster head forward
        'Lr': [],              # weighted CE loss
        'Ls': [],              # similarity loss
        'bwd': [],             # backward + step
        'total': [],           # whole batch
    },
    'gpu_memory': [],          # (epoch, step_desc, allocated_MB, reserved_MB)
    'eval_results': [],        # (epoch, acc, nmi, ari, caa)
    'per_class_acc_final': None,
    'stage_times': {},         # key stage → seconds
    'dp_final': {              # snapshot from last CRP run
        'hard_assignments': None,
        'soft_assignments': None,
        'n_occ': None,
        'subcluster_sizes': None,
        'subcluster_weights': None,   # Eq.5 weights
    },
    'head_final': {            # final head predictions on train set
        'predictions': None,
        'confidences': None,
    },
    'opt': {},                 # snapshot of key hyperparams
}

def _reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def _sample_gpu_memory(epoch, desc):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        _EXP_LOG['gpu_memory'].append({
            'epoch': epoch,
            'desc': desc,
            'allocated_MB': alloc,
            'reserved_MB': reserved,
            'peak_MB': peak,
            'timestamp': time.time(),
        })

class Cluster_projector(nn.Module):
    def __init__(self, fea_dim, cluster_num, image_centers):
        super().__init__()
        actual_fea_dim = image_centers.shape[1]
        self.layers1 = nn.Linear(actual_fea_dim, cluster_num)   # ONLY target head
        self.layers1.weight.data = 100 * image_centers.detach().clone().cuda()
        nn.init.zeros_(self.layers1.bias)
        self.cluster_num = cluster_num

    def forward(self, x):
        return self.layers1(x)                                 # no mini head any more

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            return torch.tensor(0.0, requires_grad=True, device='cuda')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)

class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing, class_weight):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing
        self.class_weight = class_weight

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        if self.class_weight is not None:
        # Class balancing weights
            weight = self.class_weight
        else:
            if self.apply_class_balancing:
                idx, counts = torch.unique(target_masked, return_counts=True)
                freq = 1 / (counts.float() / n)
                weight = torch.ones(c).cuda()
                weight[idx] = freq
            else:
                weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight=weight, reduction='mean')

        return loss

def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)

@njit
def _gibbs_sweep_numba(assignments, centers, counts, embeddings, alpha, tau, num_centers):
    """
    Numba-accelerated one sweep of CRP Gibbs sampling.
    All arrays are NumPy (float64 for centers, int64 for assignments, float64 for counts).
    """
    N, D = embeddings.shape
    for i in range(N):
        curr = assignments[i]
        counts[curr] -= 1

        # Compute log-likelihood for existing centers (cosine distance: 1 - dot -> -dot/tau for simplicity)
        log_lik = np.zeros(num_centers + 1, dtype=np.float64)
        for c in range(num_centers):
            dot = np.dot(centers[c], embeddings[i])
            log_lik[c] = dot / tau        # since both normalized, high dot = low distance
        log_lik[num_centers] = 0.0        # new table likelihood

        # Prior: CRP
        log_prior = np.log(np.append(counts[:num_centers], alpha) + 1e-10)

        probs = np.exp(log_prior + log_lik - np.max(log_prior + log_lik))
        probs /= probs.sum()
        new_table = np.searchsorted(np.cumsum(probs), np.random.random())

        if new_table == num_centers:
            # Add a new cluster
            centers[num_centers] = embeddings[i]
            counts = np.append(counts, 0.0)
            num_centers += 1

        assignments[i] = new_table
        counts[new_table] += 1

    return assignments, centers, counts, num_centers


@njit
def _gibbs_sweep_crp(assignments, centers, counts, embeddings, alpha, tau, num_centers):
    """
    One full Gibbs sweep over all samples.
    All arrays have FIXED size: centers (N+1, D), counts (N+1,).
    """
    N = embeddings.shape[0]
    D = embeddings.shape[1]
    for i in range(N):
        curr = assignments[i]
        counts[curr] -= 1
        # log-likelihood for existing centers (cosine distance: 1 - dot)
        log_lik = np.zeros(num_centers + 1)
        for c in range(num_centers):
            dot = 0.0
            for d in range(D):
                dot += centers[c, d] * embeddings[i, d]
            dist = 1.0 - dot
            log_lik[c] = -dist / tau
        log_lik[num_centers] = 0.0  # new table → distance = 0
        # CRP log-prior
        log_prior = np.zeros(num_centers + 1)
        for c in range(num_centers):
            log_prior[c] = np.log(counts[c] + 1e-8)
        log_prior[num_centers] = np.log(alpha + 1e-8)
        log_probs = log_prior + log_lik
        max_log = np.max(log_probs)
        probs = np.exp(log_probs - max_log)
        probs /= probs.sum()
        # sample table index
        r = np.random.random()
        cum = 0.0
        new_table = num_centers
        for c in range(num_centers + 1):
            cum += probs[c]
            if r < cum:
                new_table = c
                break
        if new_table == num_centers:
            # create new table if space available
            if num_centers < centers.shape[0]:
                for d in range(D):
                    centers[num_centers, d] = embeddings[i, d]
                num_centers += 1
            else:
                # fallback: choose the most probable existing center
                new_table = np.argmax(probs[:-1])
        assignments[i] = new_table
        counts[new_table] += 1
    return assignments, centers, counts, num_centers
def crp_gibbs(embeddings_np, n_sweeps=60, alpha=3.0, num_restarts=3, pca_dim=64):
    """
    Original CRP Gibbs sampler with fixed-size arrays (no dynamic allocation).
    Returns: (hard_assign, soft_assign, n_occupied, elapsed_time)
    """
    t0 = time.perf_counter()
    # PCA reduction (same seed as original)
    pca = PCA(n_components=pca_dim, random_state=42)
    embeddings_np = pca.fit_transform(embeddings_np).astype(np.float64)
    N, D = embeddings_np.shape
    # Original temperature
    tau = 2.5
    n_init_clusters = min(120, N)
    best_assign = None
    best_n_occ = 0
    total_sweeps = num_restarts * n_sweeps
    pbar = tqdm(total=total_sweeps, desc="CRP Gibbs sweeps")
    # Serial restarts to avoid multithreading memory issues
    for restart in range(num_restarts):
        # Initialize via k-means
        _, init_centers = clustering(embeddings_np, n_init_clusters)
        if isinstance(init_centers, torch.Tensor):
            init_centers = init_centers.cpu().numpy()
        num_centers = init_centers.shape[0]
        # Preallocate FIXED arrays (N+1 to allow new table without realloc)
        centers = np.zeros((N + 1, D), dtype=np.float64)
        centers[:num_centers] = init_centers
        # Initial assignments via Euclidean nearest center (as in original)
        dist2 = np.sum((embeddings_np[:, None, :] - centers[None, :num_centers, :]) ** 2, axis=-1)
        assignments = np.argmin(dist2, axis=1).astype(np.int64)
        # Fixed-size counts array
        counts = np.zeros(N + 1, dtype=np.float64)
        for idx in assignments:
            counts[idx] += 1
        # Gibbs sweeps
        for sweep in range(n_sweeps):
            assignments, centers, counts, num_centers = _gibbs_sweep_crp(
                assignments, centers, counts, embeddings_np, alpha, tau, num_centers
            )
            pbar.update(1)
        n_occ = int((counts[:num_centers] > 0).sum())
        if n_occ > best_n_occ:
            best_n_occ = n_occ
            best_assign = assignments.copy()
    pbar.close()
    # Convert to contiguous labels and soft assignments
    unique = np.unique(best_assign)
    id_map = {old: new for new, old in enumerate(unique)}
    hard_assign = np.array([id_map[a] for a in best_assign], dtype=np.int64)
    soft_assign = np.eye(len(unique), dtype=np.float32)[hard_assign]
    elapsed = time.perf_counter() - t0
    print(f"[CRP-DP] occupied mini-clusters: {len(unique)} (α={alpha}, time={elapsed:.2f}s)")
    return hard_assign, soft_assign, len(unique), elapsed

def compute_dp_subcluster_weights(hard_assignments, alpha=6.0, K_fallback=None):
    """
    Compute Eq.5 weights for each DP sub-cluster.
    w_k = ((n_k + alpha/K) / (N - 1 + alpha))^{-1}
    """
    N = len(hard_assignments)
    unique, counts = np.unique(hard_assignments, return_counts=True)
    K = K_fallback if K_fallback else len(unique)
    # Eq.5: propensity = (n_k + alpha/K) / (N - 1 + alpha)
    # weight = 1 / propensity
    smooth = alpha / max(K, 1)
    propensities = (counts.astype(np.float64) + smooth) / (N - 1 + alpha)
    weights = 1.0 / (propensities + 1e-10)
    # Normalize for readability (sum to 1)
    weights = weights / weights.sum() * len(unique)
    return dict(zip(unique, counts)), dict(zip(unique, weights))

def evaluate(cluster_predictor, cached_features, labels, log):
    """
    Optimized evaluation that skips the backbone and uses cached features.
    """
    cluster_predictor.eval()
    with torch.no_grad():
        # Process in batches to avoid VRAM spikes even for cached features
        batch_size = 1024
        all_preds = []
        all_confs = []
        for i in range(0, cached_features.size(0), batch_size):
            feat = cached_features[i:i+batch_size].cuda()
            logits = cluster_predictor(feat)
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            all_confs.append(confs.cpu())
            all_preds.append(preds.cpu())
        
        predictions = torch.cat(all_preds).numpy()
        confidences = torch.cat(all_confs).numpy()
    from models.Accuracy import cluster_accuracy
    acc, nmi, ari, ca = cluster_accuracy(labels.cpu().numpy(), predictions, verbose=False)
    ca_mean = np.mean(ca)
    log.info(f"ACC: \033[92m{acc:.4f}\033[0m, CAA: \033[92m{ca_mean:.4f}\033[0m, "f"NMI: \033[92m{nmi:.4f}\033[0m, ARI: \033[92m{ari:.4f}\033[0m")
    return acc, nmi, ari, ca_mean, ca, predictions, confidences

def compute_all_train_predictions(cluster_predictor, cached_train_feat, batch_size=1024):
    """
    Compute cluster head predictions for ALL training samples.
    Returns: predictions [N], confidences [N], softmax_probs [N, K]
    """
    cluster_predictor.eval()
    with torch.no_grad():
        all_preds, all_confs, all_probs = [], [], []
        for i in range(0, cached_train_feat.size(0), batch_size):
            feat = cached_train_feat[i:i+batch_size].cuda()
            logits = cluster_predictor(feat)
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            all_preds.append(preds.cpu())
            all_confs.append(confs.cpu())
            all_probs.append(probs.cpu())
    return (torch.cat(all_preds).numpy(),
            torch.cat(all_confs).numpy(),
            torch.cat(all_probs).numpy())

def get_class_weight(predictions, mini_assign_batch, cluster_num, minimum_ratio, maxiclustering=False):
    """
    predictions:      target-cluster hard labels (batch) → [0..K-1]
    mini_assign_batch: DP mini-cluster hard labels (batch) → [0..N_occupied-1]
    """
    # Dynamically compute a safe multiplier based on actual mini-cluster IDs
    max_mini = int(mini_assign_batch.max().item()) + 1
    multiplier = max(1000, max_mini)          # always safe, never too small
    flat = predictions * multiplier + mini_assign_batch

    minlength = cluster_num * multiplier
    cnt_matrix = torch.bincount(flat, minlength=minlength).reshape(cluster_num, multiplier).cuda().float()

    if not maxiclustering:
        weight = F.normalize(cnt_matrix, p=1, dim=0)
        weight = weight * (weight > minimum_ratio).long()
        result = torch.count_nonzero(weight, dim=1)
        result = torch.clamp(result, min=0.5)
        return 1 / (result / cnt_matrix.shape[1]), weight, cnt_matrix
    else:
        weight = F.normalize(cnt_matrix, p=1, dim=1)
        weight = weight * (weight > minimum_ratio).long()
        result = torch.count_nonzero(weight, dim=1)
        result = torch.clamp(result, min=0.5)
        return result, weight, cnt_matrix

# MODIFIED training loop (only Lr + adapted Ls; no Lm)
def train_cluster_projector_instance_cluster(opt, model, optimizer, train_loader, epoch, log, cluster_projector, global_mini_hard, global_soft):
    epoch_times = {'fwd': [], 'head': [], 'Lr': [], 'Ls': [], 'bwd': [], 'total': []}
    for i, (inputs, target, idx) in enumerate(train_loader):
        t_batch_start = time.perf_counter()
        inputs_1, inputs_2, _ = inputs
        inputs_1 = inputs_1.cuda(non_blocking=True)
        inputs_2 = inputs_2.cuda(non_blocking=True)

        if opt.backbone == 'byol':
            # BYOL fine-tuning: allow gradients for feature1, stop them for feature2
            # (target branch receives no gradient, matching old code)
            model.train()
            cluster_projector.train()
            t_fwd_start = time.perf_counter()
            with torch.no_grad():
                feature2 = model.encoder(inputs_2)
                logits2 = cluster_projector(feature2)
            feature1 = model.encoder(inputs_1)
            assert feature1.requires_grad, "feature1 must require grad for fine-tuning"
            logits1 = cluster_projector(feature1)
            t_fwd = time.perf_counter() - t_fwd_start
        else:
            # Frozen backbone: keep eval mode and no gradients
            model.eval()
            cluster_projector.train()
            t_fwd_start = time.perf_counter()
            with torch.no_grad():
                feature1 = model(inputs_1)
                feature2 = model(inputs_2)
            t_fwd = time.perf_counter() - t_fwd_start
        # ---- Cluster head forward ----
        t_head_start = time.perf_counter()
        if opt.no_target_stop_grad:
            # Allow gradients to flow back to the target branch (ablation study)
            logits2 = cluster_projector(feature2)
        else:
            # Normal: stop the grad for the target view
            with torch.no_grad():
                logits2 = cluster_projector(feature2)

        logits1 = cluster_projector(feature1)
        t_head = time.perf_counter() - t_head_start

        c = F.softmax(logits1, dim=1)
        # Similarity ablation: optionally block gradients for c
        if opt.stop_grad_similarity:
            c = c.detach()
        confidence = c.max(dim=1).values
        prediction = c.argmax(dim=1)

        # Slice global DP assignments for this batch
        mini_hard_batch = torch.from_numpy(global_mini_hard[idx.cpu().numpy()]).cuda()
        soft_mini_batch = torch.from_numpy(global_soft[idx.cpu().numpy()]).float().cuda()
        threshold = opt.threshold
        mask = confidence > threshold
        if mask.sum() == 0:
            continue

        weight, _, _ = get_class_weight(prediction, mini_hard_batch, opt.cluster_num, opt.minimum_ratio, opt.maxiclustering)
        if opt.uni_weight:
            weight = torch.ones(opt.cluster_num).cuda()

        criterion = ConfidenceBasedCE(threshold=threshold, apply_class_balancing=True, class_weight=weight)
        # ---- Lr (weighted CE) ----
        t_lr_start = time.perf_counter()
        ce_loss = criterion(logits2, logits1)   # only Lr (re-weighted)
        t_lr = time.perf_counter() - t_lr_start

        # Adapted Ls: target vs DP soft responsibilities
        # ---- Ls (similarity alignment) ----
        t_ls_start = time.perf_counter()
        sim = get_sim(c)
        other_sim = get_sim(soft_mini_batch)
        c_loss = F.mse_loss(sim, other_sim)
        t_ls = time.perf_counter() - t_ls_start

        loss = ce_loss + c_loss * opt.beta
        t_bwd_start = time.perf_counter()

        optimizer.zero_grad()
        loss.backward()
        # ===== DEBUG: verify gradients flow to encoder_k =====
        if opt.backbone == 'byol' and epoch == 1 and i == 0:
            first_conv_weight = model.encoder_k[0].weight  # 注意：ResNet 第一层可能是 conv1
            if first_conv_weight.grad is None:
                print("WARNING: encoder_k[0].weight.grad is None → backbone NOT being trained!")
            else:
                print(f"OK: encoder_k[0].weight.grad norm = {first_conv_weight.grad.norm().item():.6f}")
        # =====================================================
        optimizer.step()
        t_bwd = time.perf_counter() - t_bwd_start
        t_total = time.perf_counter() - t_batch_start
        epoch_times['fwd'].append(t_fwd)
        epoch_times['head'].append(t_head)
        epoch_times['Lr'].append(t_lr)
        epoch_times['Ls'].append(t_ls)
        epoch_times['bwd'].append(t_bwd)
        epoch_times['total'].append(t_total)
        if i == 0:
            log.info(
                f'Epoch {epoch} loss: {loss.item():.4f}  '
                f'(Lr={ce_loss.item():.4f}, Ls={c_loss.item():.4f})  '
                f't_fwd={t_fwd*1000:.1f}ms t_head={t_head*1000:.1f}ms '
                f't_Lr={t_lr*1000:.1f}ms t_Ls={t_ls*1000:.1f}ms '
                f't_bwd={t_bwd*1000:.1f}ms'
            )
        # ---- Store epoch averages ----
    for key in epoch_times:
        if epoch_times[key]:
            _EXP_LOG['batch_times'][key].append(np.mean(epoch_times[key]) * 1000)  # ms
        else:
            _EXP_LOG['batch_times'][key].append(0.0)
    if epoch == 1 or epoch % 25 == 0:
        _sample_gpu_memory(epoch, f'after_epoch_{epoch}')

def get_sim(A):
    A_norm = F.normalize(A, dim=1)
    return torch.mm(A_norm, A_norm.t())

def save_experiment_log(save_dir):
    """Save all collected experiment data for visualization."""
    # Convert numpy arrays to lists for JSON serialization where possible
    serializable = {}
    for k, v in _EXP_LOG.items():
        if k == 'dp_final':
            serializable[k] = {
                'n_occ': v['n_occ'],
                'subcluster_sizes': (
                    {str(kk): int(vv) for kk, vv in v['subcluster_sizes'].items()}
                    if v['subcluster_sizes'] else None
                ),
                'subcluster_weights': (
                    {str(kk): float(vv) for kk, vv in v['subcluster_weights'].items()}
                    if v['subcluster_weights'] else None
                ),
            }
        elif k == 'head_final':
            serializable[k] = {
                'predictions': v['predictions'].tolist() if v['predictions'] is not None else None,
                'confidences': v['confidences'].tolist() if v['confidences'] is not None else None,
            }
        elif k == 'per_class_acc_final':
            serializable[k] = v.tolist() if isinstance(v, np.ndarray) else v
        elif k == 'batch_times':
            serializable[k] = v
        elif k == 'gpu_memory':
            serializable[k] = v
        elif k == 'eval_results':
            serializable[k] = v
        elif k == 'stage_times':
            serializable[k] = v
        elif k == 'opt':
            serializable[k] = v
        else:
            serializable[k] = v
    # Save full data as pickle (for numpy arrays)
    import pickle
    pkl_path = save_dir.replace('.json', '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(_EXP_LOG, f)
    # Save serializable subset as JSON
    with open(save_dir, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"[LOG] Experiment data saved to {pkl_path} and {save_dir}")