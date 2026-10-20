import argparse
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from .byol import BYOL

class SSLWrapper(nn.Module):
    """Unifies all backbones to have a single .encoder attribute and .params() method, 
       while strictly enforcing L2 normalization for vMF space compliance."""
    def __init__(self, encoder, projector=None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        if projector is not None:
            self.add_module('projector', projector)
            
    def forward(self, x):
        out = self.encoder(x)
        if hasattr(self, 'projector') and self.projector is not None:
            out = self.projector(out)
            
        # ROBUST FIX: Ensure output is always flattened to [Batch, Features]
        if out.dim() > 2:
            out = torch.flatten(out, 1)
        # if out.size(0) > 1:
        #     out = out - out.mean(dim=0, keepdim=True)
        # out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        out = F.normalize(out, p=2, dim=1)
        return out
        
    def get_params(self):
        return self.parameters()

class SwAVProjector(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=128):
        super().__init__()
        # SwAV Projector: 2-layer MLP (3-layer variants also exist, 2 is standard for R50)
        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.projection_head(x)

# class MoCoEncoder(nn.Module):
#     def __init__(self, base_encoder, dim=2048, hidden_dim=2048, out_dim=256):
#         super().__init__()
#         self.encoder = base_encoder
#         # projection head: Linear -> ReLU -> Linear
#         self.projector = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, out_dim)
#         )
        
#     def forward(self, x):
#         feat = self.encoder(x)
#         feat = torch.flatten(feat, 1)
#         return self.projector(feat)

def load_ssl_model(opt, log):
    os.makedirs(opt.download_dir, exist_ok=True)
    opt.download_dir = os.path.join(opt.download_dir,opt.backbone)
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    if opt.backbone == 'byol':
        model = BYOL(opt)
        if opt.checkpoint and os.path.isfile(opt.checkpoint):
            log.info(f"Loading pre-trained BYOL checkpoint from: {opt.checkpoint}")
            ckpt = torch.load(opt.checkpoint, map_location='cpu', weights_only=False)
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            # Prefer loading online branch (encoder_q / projector_q) for downstream fine-tuning
            enc_k_dict = {k.replace('module.', '').replace('encoder_k.', ''): v 
                        for k, v in state_dict.items() if 'encoder_k' in k and 'projector' not in k}
            proj_k_dict = {k.replace('module.', '').replace('projector_k.', ''): v 
                        for k, v in state_dict.items() if 'projector_k' in k}
            missing_enc, _ = model.encoder_k.load_state_dict(enc_k_dict, strict=False)
            missing_proj, _ = model.projector_k.load_state_dict(proj_k_dict, strict=False)
            log.info(f"BYOL Loaded. Missing enc keys: {len(missing_enc)}, Missing proj keys: {len(missing_proj)}")
        # Return the online branch with trainable parameters (encoder_q + projector_q)
        return model, opt.fea_dim


    elif opt.backbone == 'dino':
        log.info("Loading DINOv3 via timm (with HF fallback)...")
        try:
            import timm
            model_name = getattr(opt, 'dino_model_name', 'vit_base_patch16_dinov3_qkvb')
            encoder = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                global_pool='token'
            )
            fea_dim = encoder.num_features
            class DinoFeatureExtractor(nn.Module):
                def __init__(self, model): super().__init__(); self.model = model
                def forward(self, x): return self.model(x)
            encoder = DinoFeatureExtractor(encoder)
            return SSLWrapper(encoder), fea_dim
        except ImportError:
            log.info("timm not found. Fall back to HuggingFace Transformers...")
            from transformers import AutoModel
            hf_model_name = 'facebook/dinov2-small'
            hf_model = AutoModel.from_pretrained(hf_model_name)
            fea_dim = hf_model.config.hidden_size
            class HFWrapper(nn.Module):
                def __init__(self, model): super().__init__(); self.model = model
                def forward(self, x): 
                    return self.model(x).last_hidden_state[:, 0, :]
            encoder = HFWrapper(hf_model)
            return SSLWrapper(encoder), fea_dim
        
    elif opt.backbone == 'moco':
        log.info("Loading MoCo v3 ResNet50 (Encoder Only)...")
        import torchvision.models as models
        encoder = models.resnet50()
        encoder.fc = nn.Identity()
        
        target_path = os.path.join(opt.download_dir, 'moco_v3_resnet50.pth.tar')
        if not os.path.exists(target_path):
            log.info("Downloading MoCo v3 weights...")
            url = "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar"
            torch.hub.download_url_to_file(url, target_path)
            
        ckpt = torch.load(target_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Load Encoder - safely remove base_encoder prefix 
        enc_state = {k.replace('module.base_encoder.', '').replace('base_encoder.', ''): v 
                     for k, v in state_dict.items() if 'base_encoder.' in k and 'fc' not in k}
        encoder.load_state_dict(enc_state, strict=False)
        
        # Return only the encoder and the 2048 feature dimension
        return SSLWrapper(encoder, projector=None), 2048
        
    elif opt.backbone == 'swav':
        log.info("Loading SwAV ResNet50 (Encoder Only)...")
        import torchvision.models as models
        encoder = models.resnet50()
        encoder.fc = nn.Identity()
        
        target_path = os.path.join(opt.download_dir, 'swav_800ep_pretrain.pth.tar')
        if not os.path.exists(target_path):
            log.info("Downloading SwAV full weights...")
            url = "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"
            torch.hub.download_url_to_file(url, target_path)
            
        ckpt = torch.load(target_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        
        # Isolate encoder weights: removed the restrictive 'module.' check
        enc_state = {k.replace('module.', ''): v for k, v in state_dict.items() 
                     if 'projection_head' not in k and 'prototypes' not in k}
        encoder.load_state_dict(enc_state, strict=False)
            
        # Return only the encoder and the 2048 feature dimension
        return SSLWrapper(encoder, projector=None), 2048
    else:
        raise ValueError(f"Unsupported backbone: {opt.backbone}")