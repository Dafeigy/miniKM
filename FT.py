import torch
from config import *
import numpy as np 
import cmath

def complex2argument(data:dict):
    """
    将一个sample内的复数转换成幅度和相角矩阵。一个sample是指从一根天线得到的若干个子载波组成的一维数据。
    返回两个与sample相同长度的一维矩阵。
    """
    amp =   [abs(complex(each['real'],each['image'])) for each in data]
    phase = [cmath.phase(complex(each['real'],each['image'])) for each in data]
    return amp, phase


def eliminate_319_amp_offset(raw_amp:np.ndarray):
    raw_amp[:,318,0,:] = (raw_amp[:,317,0,:] + raw_amp[:,319,0,:] )/ 2
    return raw_amp.tolist()

def process_phase(raw_phase:list)->list:
    """
    Unwrap Phase data and perform linear transform. 
    Return the preprocessed_data in list.
    """
    m = [i for i in range(-624,624,2)]
    F = np.unwrap(np.array(raw_phase)).tolist()
    k_ = (F[-1]-F[0])/(m[-1]-m[0])
    b_ = sum(F)/len(F)
    return [F[i] - k_*m[i] - b_ for i in range(len(F))]

def process_ampNphi(raw_amp:list,raw_phi:list):
    amp = eliminate_319_amp_offset(np.array(raw_amp))
    phi = process_phase(raw_phi)
    return [amp, phi]




def rec_trans(sample):
    temp = tuple(
    torch.tensor(
        process_complex(sample['ESTIMATION'][i]['LSSRS'])) for i in range(nb_gNB_TX * nb_UE_Ports))
    tensor = torch.stack(temp, dim=1)  # 组合为一个张量
    tensor = tensor.view(1, nb_subcarrier, 1, nb_gNB_TX * nb_UE_Ports)
    return tensor

def out_trans(output):
    # TODO: convert output tensor[1*n] to the ideal output and like [best_idx].
    pass


if __name__ == "__main__":
    pass
