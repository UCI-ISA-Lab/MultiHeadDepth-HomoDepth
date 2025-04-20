import argparse
import torch
import model
from matplotlib import pyplot as plt
from imageio import v2 as iio
import utils


def infer(args):
    device = torch.device(args.device)
    model_name = args.model
    data_name = args.dataset


    if model_name == 'MulH':
        in_model = model.Stereo_MulH()
    elif model_name == 'HomoDepth':
        in_model = model.HomoDepth(disp_only=True)

    if model_name == 'MulH' and data_name == 'sceneflow':
        trained_model = torch.load("ckpt/MulH_SF.pt", map_location=device, weights_only=True)
    elif model_name == 'MulH' and data_name == 'DTU':
        trained_model = torch.load("ckpt/MulH_DTU.pt", map_location=device, weights_only=True)
    elif model_name == 'MulH' and data_name == 'ADT':
        trained_model = torch.load("ckpt/MulH_ADT.pt", map_location=device, weights_only=True)
    elif model_name == 'HomoDepth' and data_name == 'DTU':
        trained_model = torch.load("ckpt/HomoDepth_DTU.pt", map_location=device, weights_only=True)
    else:
        print("There is no model matched with the dataset your required. Exit.")
        exit()

    utils.remove_module_prefix(trained_model)
    in_model.load_state_dict(trained_model['model_state_dict'])
    in_model.to(device)
    in_model.eval()

    left = utils.img2tensor(args.left_img)
    right = utils.img2tensor(args.right_img)
    image = torch.cat((left, right), dim=0)
    image = image.unsqueeze(0).to(device)

    image_tensor = image.to(device)
    output = in_model(image_tensor)

    res = output.detach().cpu().squeeze().numpy()
    plt.imshow(res, cmap='jet')
    plt.show()
    if args.save_format == 'jet':
        plt.imsave(args.save_path, res, cmap='jet')
    elif args.save_format == 'png':
        iio.imsave(args.save_path, res)

    print('Depth map is saved to ' + args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MulH',
                        choices=['MulH', 'HomoDepth'],
                        help='Name of trained model: MulH or HomoDepth')
    parser.add_argument('--dataset', '-d', type=str, default='sceneflow',
                        choices=['sceneflow', 'ADT', 'DTU'],
                        help='Name of dataset: sceneflow, ADT, or DTU')
    parser.add_argument('--left_img', '-l', type=str, required=True, help='Path to the left image')
    parser.add_argument('--right_img', '-r', type=str, required=True, help='Path to the right image')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run the model')
    parser.add_argument('--save_path', '-s', type=str, default='./res.png',
                        help='Path to save the predicted dapth map.')
    parser.add_argument('--save_format', '-f', type=str, default='jet', choices=['png', 'jet'],
                        help='Format of the saved dapth map. '
                             '\n jet: the output is visualized as color map. '
                             '\n png: the output is saved as png file with its real pixel values.')

    arguments = parser.parse_args()
    infer(arguments)







