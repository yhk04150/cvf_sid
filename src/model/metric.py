import torch
import numpy as np
import math
# from skimage.measure.simple_metrics import compare_psnr
# from skimage.measure  import compare_ssim
    
    
    
# def psnr(img, imclean):
#     img = img.mul(255).clamp(0, 255).round().div(255)
#     imclean = imclean.mul(255).clamp(0, 255).round().div(255)
#     Img = img.data.cpu().numpy().astype(np.float32)
#     Iclean = imclean.data.cpu().numpy().astype(np.float32)
#     PSNR = []
#     for i in range(Img.shape[0]):
#         ps = compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=1.0)
#         if np.isinf(ps):
#             continue
#         PSNR.append(ps)
#     return sum(PSNR)/len(PSNR)


# def ssim(img, imclean):
#     img = img.mul(255).clamp(0, 255).round().div(255)
#     imclean = imclean.mul(255).clamp(0, 255).round().div(255)
#     Img = img.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.float32)
#     Iclean = imclean.permute(0, 2, 3, 1).data.cpu().numpy().astype(np.float32)
#     SSIM = []
#     for i in range(Img.shape[0]):
#         ss = compare_ssim(Iclean[i,:,:,:], Img[i,:,:,:], multichannel =True)
#         SSIM.append(ss)
#     return sum(SSIM)/len(SSIM)
# import numpy as np
# from skimage.metrics import structural_similarity as sk_ssim

# def ssim(img, imclean):
#     """
#     Calculate the mean SSIM between two batches of images.
    
#     Args:
#         img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width)
#         imclean (torch.Tensor): Reference image tensor of shape (batch_size, channels, height, width)
    
#     Returns:
#         float: Mean SSIM score across the batch
#     """
#     # Ensure inputs are in range [0, 1], scale to [0, 255] and back to [0, 1] for consistency
#     img = img.mul(255).clamp(0, 255).round().div(255)
#     imclean = imclean.mul(255).clamp(0, 255).round().div(255)

#     # Convert PyTorch tensors to NumPy arrays, rearranging dimensions to (batch_size, height, width, channels)
#     Img = img.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
#     Iclean = imclean.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)

#     SSIM = []
#     for i in range(Img.shape[0]):
#         # Compute SSIM for each image pair in the batch
#         ss = sk_ssim(Iclean[i], Img[i], channel_axis=-1, data_range=1.0)
#         SSIM.append(ss)

#     return np.mean(SSIM)

# import torch.nn as nn
# def l1_loss(img, imgclean):
#     loss = nn.L1Loss()
#     output = loss(img, imgclean)
#     return output