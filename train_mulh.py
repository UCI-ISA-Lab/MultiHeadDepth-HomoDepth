import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
import sys

import utils
import model

sys.stdout.reconfigure(line_buffering=True)


def save_model(in_model, epoch, out_dir, optimizer, loss):
    save_name = 'epoch{}.pt'.format(epoch)
    out_path = os.path.join(out_dir, save_name)
    torch.save({
        'model_state_dict': in_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch
    }, out_path)


def valid(in_model, val_dataset, batch_size, device, samp_rate=None):
    if samp_rate is None:
        test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        subset_size = len(val_dataset) // samp_rate
        subset_sampler = SubsetRandomSampler(range(subset_size))
        test_loader = DataLoader(val_dataset, batch_size=batch_size,
                                 shuffle=False, sampler=subset_sampler)

    in_model.to(device)
    in_model.eval()
    loss1, loss2, loss3 = 0.0, 0.0, 0.0

    with torch.no_grad():
        for i, (images, disp) in enumerate(test_loader):
            images = images.to(device)
            disp = disp.to(device)
            disp_out = in_model(images)
            loss1 += utils.abs_rel_error(disp_out, disp)
            loss2 += utils.D1_metric(disp_out, disp)
            loss3 += utils.RMSE(disp_out, disp)

    loss1 /= len(test_loader)
    loss2 /= len(test_loader)
    loss3 /= len(test_loader)

    print('AbsRel: %.4f' % loss1)
    print('D1: %.4f' % loss2)
    print('RMSE: %.4f' % loss3)


def train(in_model, train_dataset, val_dataset, args):

    if args.sample_rate is None:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        subset_size = len(train_dataset) // args.sample_rate
        subset_sampler = SubsetRandomSampler(range(subset_size))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=subset_sampler)
        print('Sample rate: %d' % args.sample_rate)

    batch_num = len(train_dataset) // args.batch_size
    optimizer = optim.Adam(in_model.parameters(), lr=args.learning_rate)
    criterion = utils.SmoothLoss(1)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.checkpoint_path is not None:
        assert os.path.exists(args.checkpoint_path), f"{args.checkpoint_path} does not exist!"
        trained_model = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
        trained_model = utils.remove_module_prefix(trained_model)
        in_model.load_state_dict(trained_model['model_state_dict'])
        if 'optimizer_state_dict' in trained_model:
            in_model.to(device)  # move the model to GPU before optimizer declaration
            optimizer.load_state_dict(trained_model['optimizer_state_dict'])
        print('Trained model weights loaded')

    in_model = torch.nn.DataParallel(in_model)
    in_model.train()
    in_model.to(device)

    train_epoch = args.total_epochs - args.val_epochs
    for epo in range(train_epoch):
        running_loss = 0.0

        for i, (images, disparities) in enumerate(train_loader):

            images = images.to(device)
            disparities = disparities.to(device)
            outputs = in_model(images)
            loss = criterion(outputs, disparities)

            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Epoch %d, Loss: %.4f' % (epo, running_loss / (i + 1)))


    optimizer = optim.Adam(in_model.parameters(), lr=args.learning_rate_val)
    for epo in range(train_epoch, args.total_epochs):
        running_loss = 0.0

        for i, (images, disparities) in enumerate(train_loader):
            images = images.to(device)
            disparities = disparities.to(device)
            outputs = in_model(images)
            loss = criterion(outputs, disparities)

            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        running_loss = running_loss / (i + 1)
        print('Epoch %d, Loss: %.4f' % (epo, running_loss))
        valid(in_model, val_dataset, args.batch_size, device, args.sample_rate)
        save_model(in_model, epo, args.save_dir, optimizer, running_loss)

    print('Training finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', '-c', type=str, default=None,
                        help='Path to the trained check point. Only required for finetuning based on trained model.')
    parser.add_argument('--dataset', '-d', type=str, default='sceneflow',
                        choices=['sceneflow', 'ADT', 'DTU', 'Middlebury'],
                        help='Name of training dataset: sceneflow, ADT, DTU or Middlebury')
    parser.add_argument('--data_path', '-p', type=str,
                        help='Path to the dataset')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Device to run the model')
    parser.add_argument('--save_dir', '-s', type=str, default='ckpt',
                        help='File path to save the check point.')
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--total_epochs', '-e', type=int, default=100)
    parser.add_argument('--val_epochs', '-v', type=int, default=20)
    parser.add_argument('--sample_rate', type=int, default=None,
                        help='Sample rate of the dataset. The length of the dataset is divided by it.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=4e-4,
                        help='Learning rate of the model in training phase.')
    parser.add_argument('--learning_rate_val', '-lrv', type=float, default=4e-4,
                        help='Learning rate of the model in validation phase.')

    arguments = parser.parse_args()


    if arguments.total_epochs < arguments.val_epochs:
        print('Total number of epochs should greater than the number of validation epochs. Exit')
        exit()

    if arguments.dataset == 'sceneflow':
        train_dataset = utils.SceneFlowDataset(arguments.data_path, stereo=True)
        val_dataset = utils.SceneFlowDataset(arguments.data_path, train=False, stereo=True)
    elif arguments.dataset == 'ADT':
        train_dataset = utils.ADT(arguments.data_path, train=True)
        val_dataset = utils.ADT(arguments.data_path, train=False)
    elif arguments.dataset == 'DTU':
        train_dataset = utils.DTU(arguments.data_path, train='train', output_homo=False)
        val_dataset = utils.DTU(arguments.data_path, train='test', output_homo=False)
    elif arguments.dataset == 'Middlebury':
        train_dataset = utils.Middlebury(arguments.data_path)
        val_dataset = utils.Middlebury(arguments.data_path)
    else:
        print('Dataset not recognized. Exit')
        exit()

    input_model = model.Stereo_MulH()
    train(input_model, train_dataset, val_dataset, arguments)
