import torch
from config import *
from samples import sample



def process_complex(data:dict)->list:
    return [abs(complex(each['real'],each['image'])) for each in data]

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
    # 修改`config.py`中的配置内容。
    
    tensor = rec_trans(sample)

    print(tensor.shape)

