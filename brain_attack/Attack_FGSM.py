from __future__ import print_function
from collections import OrderedDict
from torch.utils.data import DataLoader, random_split
from dataset import CaptchaDataset
from util import characters, decode, calc_acc, use_predict, fetch_and_judge, initial, decode_target
from model import Model

import string
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image




def test( model, device, test_loader, epsilon ):

    correct = 0
    adv_examples = []
    error1 = 0
    error2 = 0

    for batch_index, (data, target, input_lengths, target_lengths) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        targets = list(decode_target(true) for true in target)
        output = model(data)
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
        output_argmax = output_argmax.cpu().numpy()
        init_pred = list(decode(pred) for pred in output_argmax)


        # Sum the initial prediction is wrong
        error1 += sum(1 for a, b in zip(targets, init_pred) if a != b)


        # Calculate the loss changed
        output_log_softmax = F.log_softmax(output, dim=-1)
        loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
        model.zero_grad()
        loss.backward()
        
        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
        output_argmax = output_argmax.cpu().numpy()
        final_pred = list(decode(pred) for pred in output_argmax)

        # Check for success
        #final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        error2 += sum(1 for a, b in zip(targets, final_pred) if a != b)
    correct = len(test_loader)*batch_size-(error2-error1)
    # Calculate final accuracy for this epsilon
    final_acc = (correct)/float(len(test_loader)*batch_size+1)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader)*batch_size+1, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

if __name__ == '__main__':
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    pretrained_model = "ctc2.pth"
    batch_size = 32
    use_cuda=True

    n_classes = len(characters)
    dataset = CaptchaDataset('all', 12)
    length = len(dataset)
    train_size, validate_size = int(0.05 * length), length - int(0.05 * length)
    train_set, valid_set = random_split(dataset, [train_size, validate_size])
    # train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=1)

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model = Model(n_classes, input_shape=(3, 64, 192)).to(device)
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    # model.eval()
    model.train()

    accuracies = []
    examples = []
    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, valid_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

