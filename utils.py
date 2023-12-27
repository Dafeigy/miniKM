import os
import numpy as np
from sklearn import metrics
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value

def cleanup():
    dist.destroy_process_group()


def load_data_person(path,demo):
    '''
    此函数用来读取文件中单个人的所有片段的数据
    '''
    action_list=['punch','kick','left','right','down']
    # label_list=[0,1,2,3,4]
    person_data=None
    person_label=None
    for index,action in enumerate(action_list):
        if(person_data is not None):
            temp_data=np.load(path+'\\'+demo+'\\'+action+'\\'+'amp.npy')
            temp_label=np.load(path+'\\'+demo+'\\'+action+'\\'+'label.npy') 
            person_data=np.concatenate((person_data,temp_data),axis=0)
            person_label=np.concatenate((person_label,temp_label),axis=0)
            # person_label=np.concatenate((person_label,len(temp_data)*[label_list[index]]),axis=0)
        else:
            person_data=np.load(path+'\\'+demo+'\\'+action+'\\'+'amp.npy')
            # person_label=len(person_data)*[label_list[index]]    
            person_label=np.load(path+'\\'+demo+'\\'+action+'\\'+'label.npy') 

    return person_data,person_label

def load_train_data(args,demo_list):
    data=None
    label=None
    for demo in demo_list:
        if data is not None:
            temp_data,temp_label=load_data_person(path=args.data_path,demo=demo)
            data=np.concatenate((data,temp_data),axis=0)
            label=np.concatenate((label,temp_label),axis=0)
        else:
            data,label=load_data_person(path=args.data_path,demo=demo)
    
    return data,label

def load_test_data(args,demo):
    action_list=['punch','kick','left','right','down']
    # label_list=[0,1,2,3,4]
    person_data=None
    person_label=None
    for index,action in enumerate(action_list):
        if(person_data is not None):
            temp_data=np.load(args.data_path+'\\'+demo+'\\'+action+'\\'+'amp.npy')
            temp_label=np.load(args.data_path+'\\'+demo+'\\'+action+'\\'+'label.npy') 
            person_data=np.concatenate((person_data,temp_data),axis=0)
            person_label=np.concatenate((person_label,temp_label),axis=0)
            # person_label=np.concatenate((person_label,len(temp_data)*[label_list[index]]),axis=0)
        else:
            person_data=np.load(args.data_path+'\\'+demo+'\\'+action+'\\'+'amp.npy')
            # person_label=len(person_data)*[label_list[index]]
            person_label=np.load(args.data_path+'\\'+demo+'\\'+action+'\\'+'label.npy') 
    repeat=3
    person_data=np.repeat(person_data,axis=0,repeats=repeat)
    person_label=np.repeat(person_label,axis=0,repeats=repeat)

    return person_data,person_label



class MyDataSet(Dataset):
    '''自定义数据集'''

    def __init__(self,data,mask,label):
        self.data = data
        self.mask = mask
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self,index):
        data=self.data[index]
        mask=self.mask[index]
        label=self.label[index]

        return data,mask,label

    @staticmethod
    def collate_fn(batch):
    # 官方实现的default_collate可以参考
    # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        datas, masks, labels = tuple(zip(*batch))
        datas = torch.stack(datas,dim=0)
        masks = torch.stack(masks,dim=0)
        labels = torch.stack(labels,dim=0)

        return datas,masks,labels



def get_metrics(y, y_pre):
	"""

	:param y:1071*6
	:param y_pre: 1071*6
	:return:
	"""
	y = y.cpu().detach().numpy()
	y_pre = y_pre.cpu().detach().numpy()
	# test_labels = dense(y)
	# test_pred = dense(y_pre)
	test_labels=np.array(y)
	test_pred=np.array(y_pre)

	acc = metrics.accuracy_score(test_labels, test_pred)
	# micro_f1 = metrics.f1_score(test_labels, test_pred, average='micro')
	# micro_precision = metrics.precision_score(test_labels, test_pred, average='micro')
	# micro_recall = metrics.recall_score(test_labels, test_pred, average='micro')
	# print(""+str(round(micro_precision,4))+"\t"+str(round(micro_recall,4))+"\t"+str(round(micro_f1,4)))
	return acc
    #micro_f1, micro_precision, micro_recall,


def process_complex(data:dict)->list:
    return [abs(complex(each['real'],each['image'])) for each in data]