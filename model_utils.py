import argparse
def set_args(description='5G classify model'):
    global logger
    #这里的一系列参数设置都是不考虑多GPU的情况
    parser=argparse.ArgumentParser(description=description)
    #数据相关的参数
    parser.add_argument('--modal',type=str,default='wireless CSI')
    parser.add_argument('--data_path',type=str,default='C:\\Users\\pan\\Downloads\\process-6-wireless')
    parser.add_argument('--output_dir',type=str,default='C:\\Users\\pan\\Downloads')
    parser.add_argument('--seq_len', type=int, default=624, help='输入的序列的长度 这里以624个子载波作为一个单位')
    parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=2, help='encoders输入 也就是数据的维度/通道数')
    parser.add_argument('--dec_in', type=int, default=2, help='decoder 输入 也就是数据的维度/通道数')
    parser.add_argument('--num_class',type=int,default=6)
    #训练相关的参数
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--batchsize',type=int,default=1)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--local_rank',type=int,default=0)
    parser.add_argument('--world_size',type=int,default=1)
    parser.add_argument('--num_workers',type=int,default=2)
    parser.add_argument('--lr',type=float,default=0.0001)
    parser.add_argument('--dist-url',default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_backend',default='nccl')
    #网络相关的参数
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=1, help='for Inception')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')


    args=parser.parse_args()

    return args