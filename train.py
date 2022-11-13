import numpy as np
import os
import json
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.allconv import AllConvNet
from models.wrn import WideResNet
from utils.synthetic_dataset import SyntheticOOD, SyntheticSoftOOD
from utils.tinyimages_cleaned import TinyImages
from losses import soft_entropy

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier with OOD data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--ood_data_path', type=str, default='data/boundary_data_alpha',
                    help="path to ood data")
parser.add_argument('--ood_criterion', type=str, default='xentropy', choices=['xentropy', 'softentropy'])
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['wrn', 'allconv'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--ood_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='snapshots', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

print(state)

torch.manual_seed(1)
np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                               trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data_in = dset.CIFAR10('data/cifarpy', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10('data/cifarpy', train=False, transform=test_transform, download=True)
    num_classes = 10
else:
    train_data_in = dset.CIFAR100('data/cifarpy', train=True, transform=train_transform)
    test_data = dset.CIFAR100('data/cifarpy', train=False, transform=test_transform)
    num_classes = 100

if '.npy' in args.ood_data_path:
    transform = trn.Compose(
        [trn.ToTensor(),
         trn.RandomHorizontalFlip(),
         trn.RandomVerticalFlip(),
         trn.Normalize(mean, std)])

    ood_data = TinyImages(args.ood_data_path, transform)

else:
    if args.ood_criterion == 'xentropy':
        ood_data = SyntheticOOD(
            root=args.ood_data_path,
            transform=trn.Compose(
                [trn.ToTensor(), trn.RandomHorizontalFlip(), trn.RandomVerticalFlip(), trn.Normalize(mean, std)]))
    elif args.ood_criterion == 'softentropy':
        ood_data = SyntheticSoftOOD(
            root=args.ood_data_path,
            transform=trn.Compose(
                [trn.ToTensor(), trn.RandomHorizontalFlip(), trn.RandomVerticalFlip(), trn.Normalize(mean, std)]))


train_loader_in = torch.utils.data.DataLoader(
    train_data_in,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

train_loader_out = torch.utils.data.DataLoader(
    ood_data,
    batch_size=args.ood_batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.prefetch, pin_memory=True)

# Create model
if args.model == 'allconv':
    net = AllConvNet(num_classes)
else:
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

optimizer = torch.optim.SGD(
    net.parameters(), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)


start_epoch = 0
checkpoints = ['_nets', '_optims']

for arch, name in [(net, '_nets'), (optimizer, '_optims')]:
    # Restore model if desired
    if args.load != '':
        for i in range(1000 - 1, -1, -1):
            model_name = os.path.join(args.load, args.dataset + '_' + args.model + name +
                                      '_epoch_' + str(i) + '.pt')
            if os.path.isfile(model_name):
                arch.load_state_dict(torch.load(model_name))
                print('Model restored! Epoch:', i)
                start_epoch = i + 1
                break
        if start_epoch == 0:
            assert False, "could not resume"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    torch.cuda.set_device(0)
    net.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


def train():
    net.train()
    loss_avg = 0.0
    ood_it = iter(train_loader_out)
    for X_in, y_in in train_loader_in:
        try:
            X_out, y_out = next(ood_it)
        except StopIteration:
            ood_it = iter(train_loader_out)
            X_out, y_out = next(ood_it)

        data = torch.cat([X_in, X_out])
        data = data.cuda()
        y_in = y_in.cuda()

        # forward
        pred = net(data)

        # backward
        optimizer.zero_grad()

        pred_in = pred[:len(X_in)]
        pred_out = pred[len(X_in):]
        loss = F.cross_entropy(pred_in, y_in)
        if args.ood_criterion == "xentropy":
            # cross-entropy from softmax distribution to uniform distribution
            loss += 0.5 * -(pred_out.mean(1) - torch.logsumexp(pred_out, dim=1)).mean()
        elif args.ood_criterion == "softentropy":
            # cross-entropy from softmax distribution to uniform distribution
            y_out = y_out.cuda()
            loss += 0.5 * soft_entropy(pred_out, y_out[:, 0].long(), y_out[:, 1].long(), y_out[0, 2])

        loss.backward()
        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg


# test function
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    state['test_loss'] = loss_avg / len(test_loader)
    state['test_accuracy'] = correct / len(test_loader.dataset)


if args.test:
    test()
    print(state)
    exit()

# Make save directory
if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

with open(os.path.join(args.save, args.dataset + '_' + args.model +
                                  '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

with open(args.save + '/params.json', 'w') as f:
    json.dump(state, f)

print('Beginning Training\n')

# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()

    train()
    test()

    # Save model
    torch.save(net.state_dict(),
               os.path.join(args.save, args.dataset + '_' + args.model + '_nets' +
                            '_epoch_' + str(epoch) + '.pt'))
    # Save optimizer
    torch.save(optimizer.state_dict(),
               os.path.join(args.save, args.dataset + '_' + args.model + '_optims' +
                            '_epoch_' + str(epoch) + '.pt'))

    # delete previous checkpoints
    for ckpt in ['_nets', '_optims']:

        prev_path = os.path.join(args.save, args.dataset + '_' + args.model + ckpt +
                                 '_epoch_' + str(epoch - 1) + '.pt')
        if os.path.exists(prev_path):
            os.remove(prev_path)

    # Show results
    with open(os.path.join(args.save, args.dataset + '_' + args.model +
                                      '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'],
        ))

    # # print state with rounded decimals
    # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100 - 100. * state['test_accuracy'])
    )
