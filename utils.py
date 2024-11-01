import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    #cmap = cmap *0.8
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        #raw_image = torch.reshape(raw_image, (3, 256, 256))
        raw_image = raw_image.permute(1, 2, 0)
        raw_image = np.uint8(raw_image.cpu().detach().numpy()*255)
        plt.figure()
        plt.imshow(raw_image)
        #change transparency of the heatmap
        plt.imshow(gcam * 255, cmap='jet', alpha=0.2)
        plt.savefig(filename)
        plt.close()
