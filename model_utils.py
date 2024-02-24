import argparse

def set_args(description='PreFN model'):
    global logger
    #这里的一系列参数设置都是不考虑多GPU的情况
    parser=argparse.ArgumentParser(description=description)
    #数据相关的参数
    parser.add_argument('--modal',type=str,default='wireless CSI')
    parser.add_argument('--subcarriers', type=int, default=624, help='输入的序列的长度 这里以624个子载波作为一个单位')
    parser.add_argument('--num_class',type=int,default=6)
    parser.add_argument('--ckpt_path',default='/Users/tiutiu/Downloads/fasternet_t0-epoch.281-val_acc1.71.9180.pth')
    #训练相关的参数
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--batchsize',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--local_rank',type=int,default=[0,1])
    parser.add_argument('--world_size',type=int,default=2)
    parser.add_argument('--num_workers',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--dist-url',default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_backend',default='nccl')
    #网络相关的参数
    parser.add_argument('--scale_factor', type=int, default=4, help='缩放因子')


    args=parser.parse_args()

    return args