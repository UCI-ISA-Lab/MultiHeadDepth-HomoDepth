import argparse
import os
import torch
from torch import nn
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
        for i, (images, homo, disp) in enumerate(test_loader):
            images = images.to(device)
            disp = disp.to(device)

            _, _, disp_pred = in_model(images)

            loss1 += utils.abs_rel_error(disp_pred, disp)
            loss2 += utils.RMSE(disp_pred, disp)
            loss3 += utils.D1_metric(disp_pred, disp)


    loss1 /= len(test_loader)
    loss2 /= len(test_loader)
    loss3 /= len(test_loader)

    print('AbsRel: %.4f' % loss1)
    print('RMSE: %.4f' % loss2)
    print('D1: %.4f' % loss3)


def train(in_model, train_dataset, val_dataset, args):
    min_mat = [0.55, -0.2, -96, -0.35, 0.85, -56, 0.9]
    max_mat = [1.05, 0.4, -15, 0.25, 1.2, 128, 1]

    if args.sample_rate is None:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        subset_size = len(train_dataset) // args.sample_rate
        subset_sampler = SubsetRandomSampler(range(subset_size))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=subset_sampler)
        print('Sample rate: %d' % args.sample_rate)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.device)
    batch_num = len(train_dataset) // args.batch_size
    optimizer = optim.Adam(in_model.parameters(), lr=args.learning_rate)
    criterion2 = utils.WMSELoss(50, device)
    criterion3 = utils.SmoothLoss(1)
    MSE = nn.MSELoss()

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

    # in_model = torch.nn.DataParallel(in_model)
    in_model.train()
    in_model.to(device)

    train_epoch = args.total_epochs - args.val_epochs
    for epo in range(train_epoch):
        running_loss1, running_loss2, running_loss3 = 0.0, 0.0, 0.0

        for i, (images, homo, disp) in enumerate(train_loader):
            images = images.to(device)
            homo = homo.to(device)
            disp = disp.to(device)

            homo_norm_pred, homo_pred, disp_pred = in_model(images)
            homo_norm = utils.homo2norm(homo, max_mat, min_mat).to(device)
            loss2 = MSE(homo_norm_pred, homo_norm)
            loss3 = criterion3(disp_pred, disp)
            loss1 = (loss2 / 2 / torch.exp(in_model.s1) + loss3 / 2 / torch.exp(in_model.s2) +
                     (in_model.s1 + in_model.s2) / 2)

            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            wmse_loss = criterion2(homo_pred, homo)
            running_loss2 += wmse_loss.item()
            # running_loss2 += loss2.item()
            running_loss3 += loss3.item()

        print('Epoch %d, L1 Loss: %.4f' % (epo, running_loss3 / (i + 1)))
        print('Epoch %d, WMSE: %.4f' % (epo, running_loss2 / (i + 1)))


    optimizer = optim.Adam(in_model.parameters(), lr=args.learning_rate_val)
    for epo in range(train_epoch, args.total_epochs):
        running_loss, running_loss2, running_loss3 = 0.0, 0.0, 0.0

        for i, (images, homo, disp) in enumerate(train_loader):
            images = images.to(device)
            homo = homo.to(device)
            disp = disp.to(device)

            homo_norm_pred, homo_pred, disp_pred = in_model(images)
            homo_norm = utils.homo2norm(homo, max_mat, min_mat).to(device)
            loss2 = MSE(homo_norm_pred, homo_norm)
            loss3 = criterion3(disp_pred, disp)
            loss1 = (loss2 / 2 / torch.exp(in_model.s1) + loss3 / 2 / torch.exp(in_model.s2) +
                     (in_model.s1 + in_model.s2) / 2)

            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            running_loss3 += loss3.item()

        running_loss = running_loss3 / (i + 1)
        print('Epoch %d, L1 Loss: %.4f' % (epo, running_loss))
        valid(in_model, val_dataset, args.batch_size, device, args.sample_rate)
        save_model(in_model, epo, args.save_dir, optimizer, running_loss)

    print('Training finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', '-c', type=str, default=None,
                        help='Path to the trained check point. Only required for finetuning based on trained model.')
    parser.add_argument('--data_path', '-p', type=str,
                        help='Path to the dataset')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help='Device to run the model')
    parser.add_argument('--save_dir', '-s', type=str, default='ckpt',
                        help='File path to save the check point.')
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--total_epochs', '-e', type=int, default=100)
    parser.add_argument('--val_epochs', '-v', type=int, default=20)
    parser.add_argument('--sample_rate', '-sr', type=int, default=None,
                        help='Sample rate of the dataset. The length of the dataset is divided by it.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=4e-4,
                        help='Learning rate of the model in training phase.')
    parser.add_argument('--learning_rate_val', '-lrv', type=float, default=4e-4,
                        help='Learning rate of the model in validation phase.')

    arguments = parser.parse_args()

    # arguments.data_path = "../data/DTU/"
    # arguments.device = 'cuda'
    # arguments.batch_size = 3
    # arguments.total_epochs = 4
    # arguments.val_epochs = 2
    # arguments.checkpoint_path = "ckpt/MulH_SF.pt"

    if arguments.total_epochs < arguments.val_epochs:
        print('Total number of epochs should greater than the number of validation epochs. Exit')
        exit()


    train_dataset = utils.DTU(arguments.data_path, train='train')
    val_dataset = utils.DTU(arguments.data_path, train='test')

    input_model = model.HomoDepth()
    train(input_model, train_dataset, val_dataset, arguments)
