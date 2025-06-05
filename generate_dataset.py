import random
import torch
import os
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from torch.nn import functional as F
from torchvision.transforms.functional import rgb_to_grayscale

def turbulence(x,K):#k=0.0025剧烈湍流；k=0.001中等湍流；k=0.00025低湍流
    # input_dtype = x.dtype
    F = torch.fft.fft2(x, dim=(2, 3))  # 转换到频域
    F = torch.fft.fftshift(F, dim=(2, 3)).cuda()
# 计算湍流退化核
    _, _, W, H = x.shape
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H))
    u = u.float().cuda()
    v = v.float().cuda()
# 计算湍流退化核
    H_turbulence = torch.exp(-K * ((u - W / 2) ** 2 + (v - H / 2) ** 2) ** (5 / 6))

# 应用退化核
    F = F * H_turbulence.unsqueeze(0).expand_as(F)  # 确保维度匹配
    f_turbulence = torch.fft.ifftshift(F, dim=(2, 3))
    f_turbulence = torch.abs(torch.fft.ifft2(f_turbulence, dim=(2, 3)))
    # f_turbulence = f_turbulence.to(input_dtype)
    return f_turbulence

def ButterworthLowPassFilter(image, d, n):
    """
    Butterworth低通滤波器
    """
    f = torch.fft.fft2(image)
    fshift = torch.fft.fftshift(f).cuda()
    s1 = torch.log(torch.abs(fshift))
    def make_transform_matrix(d):
        transform_matrix = torch.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, s1.shape))
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt

                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                transform_matrix[i, j] = 1 / (1 + (dis / d) ** (2 * n))
        return transform_matrix

    d_matrix = make_transform_matrix(d).cuda()
    new_img = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fshift * d_matrix)))
    return new_img


def generate_gaussian_noise_pt(img, sigma=10, gray_noise=0):
    """Add Gaussian noise (PyTorch version).

    Args:
        img (Tensor): Shape (b, c, h, w), range[0, 1], float32.
        scale (float | Tensor): Noise scale. Default: 1.0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = img.size()
    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(img.size(0), 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        noise_gray = torch.randn(*img.size()[2:4], dtype=img.dtype, device=img.device) * sigma / 255.
        noise_gray = noise_gray.view(b, 1, h, w)

    # always calculate color noise
    noise = torch.randn(*img.size(), dtype=img.dtype, device=img.device) * sigma / 255.

    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise
def random_generate_gaussian_noise_pt(img, sigma_range=(0, 10), gray_prob=0):
    sigma = torch.rand(
        img.size(0), dtype=img.dtype, device=img.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_gaussian_noise_pt(img, sigma, gray_noise)
def random_add_gaussian_noise_pt(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_gaussian_noise_pt(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out

def generate_poisson_noise_pt(img, scale=1.0, gray_noise=0):
    """Generate a batch of poisson noise (PyTorch version)

    Args:
        img (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        scale (float | Tensor): Noise scale. Number or Tensor with shape (b).
            Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b).
            0 for False, 1 for True. Default: 0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = img.size()
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        # round and clip image for counting vals correctly
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.
        # use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)

    # always calculate color noise
    # round and clip image for counting vals correctly
    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
    # use for-loop to get the unique values for each sample
    vals_list = [len(torch.unique(img[i, :, :, :])) for i in range(b)]
    vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
    vals = img.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(img * vals) / vals
    noise = out - img
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)
    return noise * scale
def random_generate_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0):
    scale = torch.rand(
        img.size(0), dtype=img.dtype, device=img.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_poisson_noise_pt(img, scale, gray_noise)
def random_add_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_poisson_noise_pt(img, scale_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)
def flip_tif_images(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tif'):
            # 打开图像
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img = transforms.ToTensor()(img)
            img = torch.unsqueeze(img,0)
            ori_h = img.shape[2]
            ori_w = img.shape[3]
#######################################################################################
            # ----------------------- The first degradation process ----------------------- #
            bulr_choice = random.randint(1, 2)
            if bulr_choice == 1:
                out = turbulence(img,random.uniform(25e-5,25e-4))
            else:
                out = ButterworthLowPassFilter(img, random.uniform(20,50), 2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], [0.2, 0.7, 0.1])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, 1.5)
            elif updown_type == 'down':
                scale = np.random.uniform(0.15, 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = 0.4
            if np.random.uniform() < 0.5:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=[1, 30], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=[0.05, 3],
                    gray_prob = gray_noise_prob,
                    clip=True,
                    rounds=False)
            # ----------------------- The second degradation process ----------------------- #
            # blur
            bulr_choice = random.randint(1, 2)
            if bulr_choice == 1:
                out = turbulence(out, random.uniform(25e-5, 25e-4))
            else:
                out = ButterworthLowPassFilter(out, random.uniform(20, 50), 2)
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], [0.3, 0.4, 0.3])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, 1.2)
            elif updown_type == 'down':
                scale = np.random.uniform(0.3, 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(int(ori_h // scale), int(ori_w // scale)), mode=mode)
            # add noise
            gray_noise_prob = 0.4
            if np.random.uniform() < 0.5:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=[0.3, 1.2], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=[0.05, 2.5],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(int(ori_h // SRscale), int(ori_w // SRscale)), mode=mode)
#######################################################################################
            output_path = os.path.join(output_folder, filename)
            save_image(out, output_path)

    print("所有文件已处理完成")

SRscale = 4
input_folder = r'F:\test3\data\HR\test'
output_folder = r'F:\test3\data\LR\test\_4'
flip_tif_images(input_folder, output_folder)
