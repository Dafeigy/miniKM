import torch
from config import *
from samples import sample



def process_complex(data:dict)->list:
    return [abs(complex(each['real'],each['image'])) for each in data]

if __name__ == "__main__":
    # 修改`config.py`中的配置内容。
    temp = tuple(
    torch.tensor(
        process_complex(sample['eSTIMATION'][i]['lSSRS'])) for i in range(nb_gNB_TX * nb_UE_Ports))

    tensor = torch.stack(temp, dim=1)  # 组合为一个张量
    tensor = tensor.view(1, nb_subcarrier, 1, nb_gNB_TX * nb_UE_Ports)  # 调整尺寸为[624, 1, 2]

    print(tensor.shape)

