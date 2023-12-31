'''
Modified based on https://github.com/DebangLi/one-pixel-attack-pytorch
'''

import os
import numpy as np

import argparse

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

from differential_evolution import differential_evolution


from util import *


CKPT_PATH = 'models/pneu_model.ckpt'
IMG_PATH = "./img"
IMG_PATH_NORM = './img/0/'
IMG_PATH_PNEU = './img/1/'
BI_ClASS_NAMES = ['Normal', 'Pneumonia']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='One pixel attack with PyTorch')
parser.add_argument('--pixels', default=20, type=int, help='The number of pixels that can be perturbed.')
# parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--maxiter', default=100, type=int, help='The maximum number of iteration in the DE algorithm.')
parser.add_argument('--popsize', default=400, type=int, help='The number of adverisal examples in each iteration.')
# parser.add_argument('--popsize', default=16, type=int, help='The number of adverisal examples in each iteration.')
parser.add_argument('--samples', default=100, type=int, help='The number of image samples to attack.')
# parser.add_argument('--samples', default=4, type=int, help='The number of image samples to attack.')
parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

#args = parser.parse_args("--verbose --targeted")

args = parser.parse_args()


def perturb_image(xs, img):
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)

    count = 0
    for x in xs:
        pixels = np.split(x, len(x) / 5)

        for pixel in pixels:
            x_pos, y_pos, r, g, b = pixel
            imgs[count, 0, x_pos, y_pos] = (r / 255.0 - 0.4914) / 0.2023
            imgs[count, 1, x_pos, y_pos] = (g / 255.0 - 0.4822) / 0.1994
            imgs[count, 2, x_pos, y_pos] = (b / 255.0 - 0.4465) / 0.2010
        count += 1

    return imgs


def predict_classes(xs, img, target_calss, net, minimize=True):
    imgs_perturbed = perturb_image(xs, img.clone())
    input = Variable(imgs_perturbed).to(device)
    predictions = F.softmax(net(input)).data.cpu().numpy()[:, target_calss]

    return predictions if minimize else 1 - predictions


def attack_success(x, img, target_class, net, targeted_attack=False, verbose=False):
    attack_image = perturb_image(x, img.clone())
    input = Variable(attack_image).to(device)
    confidence = F.softmax(net(input)).data.cpu().numpy()[0]
    predicted_class = np.argmax(confidence)

    if (verbose):
        print("Confidence: %.4f" % confidence[target_class])
    if (targeted_attack and predicted_class == target_class) or (
            not targeted_attack and predicted_class != target_class):
        return True


def attack(img, label, net, target=None, pixels=1, maxiter=75, popsize=400, verbose=False):
    # img: 1*3*W*H tensor
    # label: a number

    targeted_attack = target is not None
    target_class = target if targeted_attack else label

    bounds = [(0, 224), (0, 224), (0, 255), (0, 255), (0, 255)] * pixels

    popmul = max(1, popsize / len(bounds))

    predict_fn = lambda xs: predict_classes(
        xs, img, target_class, net, target is None)
    callback_fn = lambda x, convergence: attack_success(
        x, img, target_class, net, targeted_attack, verbose)

    inits = np.zeros([int(popmul * len(bounds)), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i * 5 + 0] = np.random.random() * 224
            init[i * 5 + 1] = np.random.random() * 224
            init[i * 5 + 2] = np.random.normal(128, 127)
            init[i * 5 + 3] = np.random.normal(128, 127)
            init[i * 5 + 4] = np.random.normal(128, 127)

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul, mutation=(0.5, 1),
                                           recombination=0.7, atol=-1, callback=callback_fn, polish=True, init=inits)

    attack_image = perturb_image(attack_result.x, img)
    attack_var = Variable(attack_image).to(device)
    predicted_probs = F.softmax(net(attack_var)).data.cpu().numpy()[0]

    predicted_class = np.argmax(predicted_probs)

    if (not targeted_attack and predicted_class != label) or (targeted_attack and predicted_class == target_class):
        return 1, attack_result.x.astype(int)
    return 0, [None]


def attack_all(net, loader, pixels=1, targeted=False, maxiter=75, popsize=400, verbose=False):
    correct = 0
    success = 0

    for batch_idx, (input, target) in enumerate(loader):

        img_var = Variable(input).to(device)
        torch.no_grad()

        prior_probs = F.softmax(net(img_var), dim=1)  # 修改概率标签!!!!!!!!!!!!!!
        _, indices = torch.max(prior_probs, 1)

        if target[0] != indices.data.cpu()[0]:
            continue

        correct += 1
        target = target.numpy()

        targets = [None] if not targeted else range(2)

        for target_calss in targets:
            if (targeted):
                if (target_calss == target[0]):
                    continue

            flag, x = attack(input, target[0], net, target_calss, pixels=pixels, maxiter=maxiter, popsize=popsize,
                             verbose=verbose)

            success += flag
            if (targeted):
                success_rate = float(success) / (9 * correct)
            else:
                success_rate = float(success) / correct

            if flag == 1:
                print("success rate: %.4f (%d/%d) [(x,y) = (%d,%d) and (R,G,B)=(%d,%d,%d)]" % (success_rate, success, correct, x[0], x[1], x[2], x[3], x[4]))

        if correct == args.samples:
            break

    return success_rate


def main():
    print("==> Loading data and model...")


    model = loadPneuModel(CKPT_PATH)

    fileFolder = IMG_PATH


    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    image_dataset = datasets.ImageFolder(fileFolder, data_transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=0)

    print("==> Starting attck...")

    results = attack_all(model, dataloader, pixels=args.pixels, targeted=args.targeted, maxiter=args.maxiter,
                         popsize=args.popsize, verbose=args.verbose)
    print("Final success rate: %.4f" % results)


if __name__ == '__main__':
    main()