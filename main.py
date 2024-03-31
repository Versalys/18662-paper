from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
# from hids_model import hidsmodel
from os import listdir


NORMAL_ROOT = 'data/ADFA-LD/Training_Data_Master/'
ATTACK_ROOT = 'data/ADFA-LD/Attack_Data_Master/'
ATTACKS = ['AddUser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
NUM_FEATURES = 265


def assemble_attack_data(arch_root=ATTACK_ROOT):
    data = {}
    for attack in ATTACKS:
        root = arch_root + attack + '/'
        attack_data = []
        for directory in listdir(root):
            for filename in listdir(root+directory):
                with open(root+directory+'/'+filename, 'r') as f:
                    attack_data.append([int(call) for call in f.read().split()])
        data[attack] = attack_data
    return data


def assemble_normal_data(root=NORMAL_ROOT):
    data = []
    for filename in listdir(root):
        with open(root+filename) as f:
            data.append([int(call) for call in f.read().split()])
    return data


def create_dataloader(attack_data, normal_data, excl={}):
    all_attack = []
    for name, instances in attack_data.items():
        if name in excl:
            continue
        all_attack.extend(zip(instances, [1 for _ in range(len(instances))]))

    all_normal = list(zip(normal_data, [0 for _ in range(len(normal_data))]))

    full_data = all_attack + all_normal
    print(full_data[-1])
    return DataLoader(full_data, batch_size=1, shuffle=True)


def train_model():
    model = hidsmodel()



if __name__ == '__main__':
    n_data = assemble_normal_data()
    a_data = assemble_attack_data()
    dataloader = create_dataloader(a_data, n_data)
    for x, y in dataloader:
        print(y)
