from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from hids_model import HidsModel
from os import listdir


NORMAL_ROOT = 'data/ADFA-LD/Training_Data_Master/'
ATTACK_ROOT = 'data/ADFA-LD/Attack_Data_Master/'
ATTACKS = ['AddUser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
NUM_FEATURES = 341


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


def create_dataloader(attack_data, normal_data, excl=set()):
    all_attack = []
    for name, instances in attack_data.items():
        if name in excl:
            continue
        all_attack.extend(zip(instances, [1 for _ in range(len(instances))]))

    all_normal = list(zip(normal_data, [0 for _ in range(len(normal_data))]))

    full_data = all_attack + all_normal
    return DataLoader(full_data, batch_size=1, shuffle=True)


def make_one_hot(n, num_classes):
    print(n)
    tensor = F.one_hot(n, num_classes)
    return tensor.float()


def train_model(dataloader, n_epochs):
    model = HidsModel(NUM_FEATURES, 128, 1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epochs):
        cumulative_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            hot_x = make_one_hot(torch.LongTensor(x), NUM_FEATURES)
            print(hot_x)
            output = model(hot_x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
        print(f'Epoch [{epoch}] : {cumulative_loss}')


def main():
    attack = assemble_attack_data()
    normal = assemble_normal_data()
    dataloader = create_dataloader(attack, normal)
    train_model(dataloader, 15)


if __name__ == '__main__':
    main()
