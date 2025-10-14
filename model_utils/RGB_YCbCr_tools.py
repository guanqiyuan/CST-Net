import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math

def rgb_to_ycbcr(image):
    r"""Convert an RGB image to YCbCr.

    Args:
        image (torch.Tensor): RGB Image to be converted to YCbCr.

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    delta = .5
    y: torch.Tensor = .299 * r + .587 * g + .114 * b
    cb: torch.Tensor = (b - y) * .564 + delta
    cr: torch.Tensor = (r - y) * .713 + delta
    return torch.stack((y, cb, cr), -3)

def ycbcr_to_rgb(image):
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape :math:`(*, 3, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = ycbcr_to_rgb(input)  # 2x3x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    y = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta

    r = y + 1.403 * cr_shifted
    g = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3)

def combine_YCbCr_and_RGB(rgb_image, ycbcr_image):
    rgb_to_ycbcr_image = rgb_to_ycbcr(rgb_image)
    rgb_to_ycbcr_image[:, 0, :, :] = ycbcr_image[:, 0, :, :]
    rgb_image = ycbcr_to_rgb(rgb_to_ycbcr_image)
    return rgb_image, ycbcr_image



class RGBToYCbCrTransform(nn.Module):
    def __init__(self, expansion_factor, alpha_R, alpha_G, alpha_B):
        super(RGBToYCbCrTransform, self).__init__()

        self.expand = expansion_factor
        self.alpha_R = alpha_R
        self.alpha_G = alpha_G
        self.alpha_B = alpha_B
        
        self.mlp = nn.Sequential(
            nn.Linear(3, self.expand),
            nn.ReLU(),
            nn.Linear(self.expand, 3)
        )

    def forward(self, image):
        R = image[:, 0, :, :]
        G = image[:, 1, :, :]
        B = image[:, 2, :, :]

        delta = 0.5
        Y = .299 * R * self.alpha_R + .587 * G * self.alpha_G + .114 * B * self.alpha_B
        Cb = (B * self.alpha_B - Y) * .564 + delta
        Cr = (R * self.alpha_R - Y) * .713 + delta

        ycbcr_stack = torch.stack([Y, Cb, Cr], dim=1)  # [B, 3, H, W]
        output_ycbcr = self.mlp(ycbcr_stack.view(-1, 3))
        output_ycbcr = output_ycbcr.view(image.shape)

        return output_ycbcr
    

class YCbCrToRGBTransform(nn.Module):
    def __init__(self, expansion_factor, beta_Y, beta_Cb, beta_Cr):
        super(YCbCrToRGBTransform, self).__init__()

        self.expand = expansion_factor
        self.beta_Y = beta_Y
        self.beta_Cb = beta_Cb
        self.beta_Cr = beta_Cr

        self.mlp = nn.Sequential(
            nn.Linear(3, self.expand),
            nn.ReLU(),
            nn.Linear(self.expand, 3)
        )

    def forward(self, image):
        Y = image[:, 0, :, :]
        Cb = image[:, 1, :, :]
        Cr = image[:, 2, :, :]

        delta = 0.5
        R = Y * self.beta_Y + 1.403 * (Cr * self.beta_Cr - delta)
        G = Y * self.beta_Y - 0.714 * (Cr * self.beta_Cr - delta) - 0.344 * (Cb * self.beta_Cb - delta)
        B = Y * self.beta_Y + 1.773 * (Cb * self.beta_Cb - delta)

        rgb_stack = torch.stack([R, G, B], dim=1)  # [B, 3, H, W]
        output_rgb = self.mlp(rgb_stack.view(-1, 3))
        output_rgb = output_rgb.view(image.shape)

        return output_rgb