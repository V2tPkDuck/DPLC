import argparse
import os
from models.util import get_dataset, logger
def main_cfg():
    
    parser = argparse.ArgumentParser('Default arguments for training of different methods')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--num_workers', type=int, default=2, help='num of workers to use')
    
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--imb_ratio', type=int, default=10, help='imbalance ratio')
    
    parser.add_argument('--crp', type=str, default='crp', choices=['crp', 'kmeans'], help='CRP vs K-means')
    parser.add_argument('--uni_weight', action='store_true', default=False,
    help='if set, use uniform class weights (w_k=1) for ablation')
    parser.add_argument('--no_target_stop_grad', action='store_true', help='If set, do NOT stop gradient on target head output (logits2)')
    parser.add_argument('--stop_grad_similarity', action='store_true', help='If set, stop gradient on predicted probabilities c when computing similarity loss')
    
    parser.add_argument('--backbone', type=str, default='byol', choices=['byol', 'swav', 'moco', 'dino'], help='SSL backbone to use')
    parser.add_argument('--download_dir', type=str, default="...", help='Directory to save downloaded weights') # 
    parser.add_argument('--cached_feature', type=str, default="")

    parser.add_argument('--resume_epoch', type=int, default=0, help='resume epoch')
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--data_folder', type=str, default="", help='path to custom dataset')
    
    parser.add_argument('--resume', default=False, type=bool, help='if resume training')
    parser.add_argument('--train_from_start', default=True, type=bool, help='if resume training')
    parser.add_argument('--train_from_resume', default=False, type=bool, help='if resume training')
    parser.add_argument('--other_cluster_num', type=int, default=200)
    parser.add_argument('--minimum_ratio', type=float, default=0.15)
    parser.add_argument('--alpha', type=float, default=2.0, help='trade-off parameter')
    parser.add_argument('--beta', type=float, default=0.7, help='trade-off parameter')
    parser.add_argument('--threshold', type=float, default=0.95, help='self-training threshold')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='base learning rate')
    parser.add_argument('--maxiclustering', type=bool, default=False, help='maxiclustering mode')
    
    
    # distributed training
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_devices', type=int, default=1, help='number of devices to use')
    
    # optimiaztion
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--momentum_base', type=float, default=0.996, help='momentum')
    parser.add_argument('--momentum_max', type=float, default=1, help='momentum')
    parser.add_argument('--momentum_increase', help='momentum_increase', action='store_true')
    parser.add_argument('--amp', action='store_true', help='amp')
    # parser.add_argument('--encoder_name', type=str, default='bigresnet18', help='the type of encoder')
    parser.add_argument('--exclude_bias_and_bn', help='exclude_bias_and_bn', action='store_true')
    
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    
    parser.add_argument('--acc_grd_step', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=10, help='lr warmup epochs')
    parser.add_argument('--dist', action='store_false', help='use for clustering')
    parser.add_argument('--hidden_size', help='hidden_size', type=int, default=4096)
    parser.add_argument('--syncbn', help='syncbn', action='store_false')
    parser.add_argument('--shuffling_bn', help='shuffling_bn', action='store_false')
    parser.add_argument('--temperature', help='temperature', type=float, default=0.5)
    parser.add_argument('--test_resized_crop', action='store_true', help='imagenet test transform')
    parser.add_argument('--resized_crop_scale', type=float, default=0.08, help='randomresizedcrop scale')
    parser.add_argument('--use_gaussian_blur', action='store_true', help='use_gaussian_blur')
    parser.add_argument('--learning_eta_min', type=float, default=0, help='minimum learning rate')# 0.
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1)
    parser.add_argument('--lr_decay_milestone', nargs='+', type=int, default=[60, 80])
    parser.add_argument('--step_lr', action='store_true', help='step_lr')
    parser.add_argument('--fix_predictor_lr', help='fix the lr of predictor', action='store_true')
    parser.add_argument('--lambda_predictor_lr', help='fix the lr of predictor', type=float, default=10)
    parser.add_argument('--momentum', type=float, default=0.9, help='lr momemtum')
    parser.add_argument('--scheduler', type=str, default='cosine', help='scheduler')
    args = parser.parse_args()
    if args.backbone == 'byol':
        args.batch_size = 512
        args.save_freq=100
    if args.dataset == 'cifar10':
        args.encoder_name = 'bigresnet18'
    elif args.dataset == 'cifar20':
        args.encoder_name = 'bigresnet18'
    elif args.dataset == 'stl10':
        args.encoder_name = 'resnet18'
    elif args.dataset == 'tiny-imagenet-200':
        args.encoder_name = 'resnet18'
    elif args.dataset == 'imagenet-lt':
        args.encoder_name = 'resnet50'
    elif args.dataset == 'imagenet100':
        args.encoder_name = 'resnet50'
    elif args.dataset == 'imagenet50':
        args.encoder_name = 'resnet50'
        
    args = get_dataset_cfg(args)
    return args
def get_dataset_cfg(opt):
    if opt.dataset == 'cifar10':
        opt.data_folder = "..."
        opt.save_dir = f".../results/{opt.dataset}/imb_{opt.imb_ratio}/{opt.backbone}/{opt.crp}/"
        opt.checkpoint = f".../checkpoints/model_1000.pt"
        opt.cached_feature = f".../{opt.dataset}/{opt.backbone}/"
        if opt.backbone == 'dino':
            opt.img_size = 224
        else:
            opt.img_size = 32
        opt.num_cluster = 10
        opt.encoder_name = "bigresnet18"
        opt.fea_dim = 512
        opt.max_num = 5000
        
    elif opt.dataset == 'cifar20':
        opt.data_folder = ""
        opt.save_dir = f""
        opt.checkpoint = f""
        opt.img_size = 32
        opt.num_cluster = 20
        opt.encoder_name = 'bigresnet18'
        opt.fea_dim = 512
        opt.max_num = 2500

    elif opt.dataset == 'stl10':
        opt.data_folder = ""
        opt.save_dir = f""
        opt.checkpoint = f""
        opt.img_size = 96
        opt.num_cluster = 10
        opt.encoder_name = 'resnet18'
        opt.fea_dim = 256
        opt.max_num = 1300

    elif opt.dataset == 'tiny-imagenet-200':
        opt.data_folder = ""
        opt.save_dir = f""
        opt.checkpoint = f""
        opt.img_size = 64  # 96
        opt.num_cluster = 200
        opt.encoder_name = 'resnet18'
        opt.test_resized_crop = True
        opt.fea_dim = 512
        opt.max_num = 500

    elif opt.dataset == 'imagenet-lt':
        opt.data_folder = ""
        opt.save_dir = f""
        opt.checkpoint = f""
        opt.img_size = 224
        opt.num_cluster = 1000
        opt.encoder_name = 'resnet50'
        opt.test_resized_crop = True
        opt.fea_dim = 512
        
    elif opt.dataset == 'imagenet100':
        opt.data_folder = ""
        opt.save_dir = f""
        opt.checkpoint = f""
        opt.img_size = 224
        opt.num_cluster = 100
        opt.encoder_name = 'resnet50'
        opt.test_resized_crop = True
        opt.fea_dim = 2048
        opt.max_num = 1300

    elif opt.dataset == 'imagenet50':
        opt.data_folder = ""
        opt.save_dir = f""
        opt.checkpoint = f""
        opt.img_size = 224
        opt.num_cluster = 50
        opt.encoder_name = 'resnet50'
        opt.test_resized_crop = True
        opt.fea_dim = 2048
        opt.max_num = 1300
    else:
        print("unknown dataset")
    opt.cluster_num = opt.num_cluster
    if opt.save_dir and not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)
    return opt
    
def get_config(byol=False):
    opt = main_cfg()
    return opt
