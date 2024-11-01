import torch.nn as nn
import os.path as osp
import numpy as np
import torch
from torchvision import models, transforms
from Grad_cam import BackPropagation,GradCAM
from torch.utils.data import Dataset, DataLoader
import os
import natsort
from train_pred import pred_val
from dataloader import CustomGradDataset
from utils import save_gradient, save_gradcam
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Retina_test_grad')
    parser.add_argument("--seed", default=1657, help="Seed number")
    parser.add_argument("--batch_size", default=8, help="Batch_size")
    parser.add_argument("--test_data_path", default="C:/Users/Jen/Desktop/Retina/test_big/test1", help="Train data path")
    parser.add_argument("--test_labels", default="C:/Users/Jen/Desktop/Original/DSK/Desktop/HC/test1_label.txt",
                        help="Train label path")
    parser.add_argument("--resize_size", default=256, help="Image resize size")
    parser.add_argument("--model_file", default='C:/Users/Jen/Desktop/Retina/models/model_pt_rs_0_58.pt',
                        help="Model saving path")
    parser.add_argument("--target_layer", default='layer4.1.conv1', help="Target layer")
    parser.add_argument("--save_img_path", default='C:/Users/Jen/Desktop/Retina/gradcam', help="Path to save grad img")
    parser.add_argument("--topk", default=2, help="")
    parser.add_argument("--gradcam", default=True, help="True for saving gradcam images, False for test results")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    # Define model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    # Load fine-tune layer weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(args.model_file))
    model=model.cuda()

    transform_img= transforms.Compose([
        transforms.Resize(args.resize_size),    #testset2: 360 testset3: 320
        transforms.CenterCrop(args.resize_size),
        transforms.ToTensor(),
        ])

    # Test data results
    if args.gradcam == False:
        test_dataset = CustomGradDataset(args.test_labels, args.test_data_path, transform=transform_img, gradcam=args.gradcam)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        all_pred = pred_val(test_loader, model)
        all_imgs = os.listdir(args.test_data_path)
        total_imgs = natsort.natsorted(all_imgs)

        # Print predicted labels
        for i in range(np.size(total_imgs)):
            print(f'{total_imgs[i]},{int(all_pred[i])}')

    # Gradcam save image
    else:
        test_gradcam_dataset = CustomGradDataset(args.test_labels, args.test_data_path, transform=transform_img, gradcam=args.gradcam)
        test_gradcam_loader = DataLoader(test_gradcam_dataset, batch_size=args.batch_size, shuffle=False)

        for i, (inputs, labels) in enumerate(test_gradcam_loader, 0):
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
            images = inputs
            bp = BackPropagation(model=model)
            probs, ids = bp.forward(images)  # sorted

            print("Grad-CAM/Guided Backpropagation/Guided Grad-CAM:")

            gcam = GradCAM(model=model)
            _ = gcam.forward(images)

            for k in range(args.topk):
                # Grad-CAM
                gcam.backward(ids=ids[:, [k]])
                regions = gcam.generate(target_layer=args.target_layer)
                for j in range(len(images)):
                    # Grad-CAM
                    save_gradcam(
                        filename=osp.join(
                            args.save_img_path,
                            "{}-gradcam.png".format(
                                j+i*args.batch_size
                            ),
                        ),
                        gcam=regions[j, 0],
                        raw_image=images[j],
                    )

