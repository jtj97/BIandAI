from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from dataset import CaptchaDataset
from model import Model
from status import username, password
from util import characters, calc_acc, use_predict, fetch_and_judge, initial

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(1)


def train(model, optimizer, epoch, dataloader):
    model.train()
    loss_mean = 0
    acc_mean = 0
    with tqdm(dataloader) as pbar:
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            optimizer.zero_grad()
            output = model(data)

            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            loss = loss.item()
            acc = calc_acc(target, output, use_cuda)

            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc

            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean

            pbar.set_description(f'Epoch: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


def valid(model, epoch, dataloader):
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss = loss.item()
            acc = calc_acc(target, output, use_cuda)

            loss_sum += loss
            acc_sum += acc

            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)

            pbar.set_description(f'Test : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')


if __name__ == '__main__':
    # 批
    batch_size = 128

    n_classes = len(characters)
    dataset = CaptchaDataset('all', 12)
    length = len(dataset)
    train_size, validate_size = int(0.8 * length), length - int(0.8 * length)
    train_set, valid_set = random_split(dataset, [train_size, validate_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=1)

    image, label, input_size, output_shape = dataset[0]
    # channel-3 height-20 width-60
    print('模型输出样例')
    print(image.shape, label, input_size, output_shape)

    # 模型定义
    model = Model(n_classes, input_shape=(3, 64, 192))
    if use_cuda:
        model = model.cuda()

    # 测试模型输出
    inputs = torch.zeros((1, 3, 64, 192))
    if use_cuda:
        inputs = inputs.cuda()
    outputs = model(inputs)
    print(outputs.shape)

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, amsgrad=True)
    epochs = 30
    for epoch in range(1, epochs + 1):
        train(model, optimizer, epoch, train_loader)
        valid(model, epoch, valid_loader)

    optimizer = torch.optim.Adam(model.parameters(), 1e-4, amsgrad=True)
    epochs = 15
    for epoch in range(1, epochs + 1):
        train(model, optimizer, epoch, train_loader)
        valid(model, epoch, valid_loader)

    torch.save(model, 'ctc.pth')
