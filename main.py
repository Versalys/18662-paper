from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from hids_model import HidsModel
from os import listdir
from tqdm import tqdm


NORMAL_ROOT = 'data/ADFA-LD/Training_Data_Master/'
ATTACK_ROOT = 'data/ADFA-LD/Attack_Data_Master/'
ATTACKS = ['AddUser', 'Hydra_FTP', 'Hydra_SSH', 'Java_Meterpreter', 'Meterpreter', 'Web_Shell']
NUM_FEATURES = 341
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU acceleration

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
    tensor = F.one_hot(n, num_classes)
    return torch.unsqueeze(tensor.float(), dim=0).to(DEVICE)


def save_model(model):
    torch.save(model, './model.pt')

def load_model():
    model = torch.load('./model.pt')
    return model


def train_model(dataloader, n_epochs):
    print(f"Training with device: '{DEVICE}'")
    model = HidsModel(NUM_FEATURES, 256, 1)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epochs):
        cumulative_loss = 0
        max_loss = -float('inf')
        min_loss = float('inf')
        for x, y in tqdm(dataloader, desc=f'Epoch [{epoch}]'):
            optimizer.zero_grad()
            hot_x = make_one_hot(torch.LongTensor(x), NUM_FEATURES)
            # print(hot_x.shape)
            output = model(hot_x)[0,-1]
            # print(output.shape)
            # print('===', output)
            loss = criterion(output, y.float())
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            max_loss = max(max_loss, loss.item())
            min_loss = min(min_loss, loss.item())
        cumulative_loss /= len(dataloader)
        print(f'Epoch [{epoch}] : <{cumulative_loss}> | Max Loss: {max_loss} | Min Loss: {min_loss}')
    return model


def test_model(model, dataloader):
    model.eval()
    correct = 0
    for x, y in tqdm(dataloader, desc='<Testing model>'):
        hot_x = make_one_hot(torch.LongTensor(x), NUM_FEATURES)
        output = model(hot_x)[-1]
        choice = 0 if output <= .5 else 1
        if choice == y:
            correct += 1
    accuracy = correct / len(dataloader)
    print(f"Accuracy: {round(accuracy*100, 3)}%")


def main():
    attack = assemble_attack_data()
    normal = assemble_normal_data()
    dataloader = create_dataloader(attack, normal)

    model = train_model(dataloader, 5)
    save_model(model)

    # model = load_model()

    test_model(model, dataloader)


if __name__ == '__main__':
    main()
