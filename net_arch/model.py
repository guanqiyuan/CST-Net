import torch
import torch.nn as nn
import torch.nn.functional as F
import thop
from einops import rearrange


# import sys
# sys.path.append(".")

from model_utils.mlp import INR
from model_utils.RGB_YCbCr_tools import RGBToYCbCrTransform, YCbCrToRGBTransform
from model_utils.utils import LayerNorm, Upsample, Downsample, OverlapPatchEmbed

global alpha_r
global alpha_g
global alpha_b
global beta_y
global beta_cb
global beta_cr
alpha_r = nn.Parameter(torch.tensor(1.0))
alpha_g = nn.Parameter(torch.tensor(1.0))
alpha_b = nn.Parameter(torch.tensor(1.0))
beta_y = nn.Parameter(torch.tensor(1.0))
beta_cb = nn.Parameter(torch.tensor(1.0))
beta_cr = nn.Parameter(torch.tensor(1.0))


class SE(nn.Module):
    def __init__(self, in_channels, ration=16):
        super(SE, self).__init__()

        self.in_ch = in_channels
        self.ration = ration
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Conv2d(in_channels=self.in_ch, out_channels=ration, kernel_size=1),
                                nn.GELU(),
                                nn.Conv2d(in_channels=self.ration, out_channels=self.in_ch, kernel_size=1),
                                nn.Sigmoid(),
                                )
    def forward(self, x):
        f = self.se(x)
        out = x + (f * x)
        return  out


class SERB(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(SERB, self).__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels

        self.backbone = nn.Sequential(nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=1),
                                      nn.InstanceNorm2d(self.out_ch),
                                      nn.GELU(),
                                      nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=1),
                                      nn.InstanceNorm2d(self.out_ch),
                                      )
        self.res = nn.Sequential(nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=1),
                                 nn.InstanceNorm2d(self.out_ch),
                                 SE(in_channels=self.out_ch),
                                 )
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.backbone(x)
        res = self.res(x)
        out = self.act(x1 + res)
        return out


class MSFN(nn.Module):
    def __init__(self, in_channels, ffn_expansion_factor=1, bias=False):
        super(MSFN, self).__init__()

        self.in_ch = in_channels

        hidden_features = int(self.in_ch * ffn_expansion_factor)

        self.project_in = nn.Conv2d(self.in_ch, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, self.in_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)
        x = self.project_out(x)
        return x


class CPSB(nn.Module):
    def __init__(self, dim, LayerNorm_type, bias=False) -> None:
        super(CPSB, self).__init__()

        self.dim = dim
        self.layernorm_type = LayerNorm_type

        self.norm1 = LayerNorm(dim=self.dim, LayerNorm_type=self.layernorm_type)
        self.serb = SERB(in_channels=self.dim, out_channels=self.dim)

        self.norm2 = LayerNorm(dim=self.dim, LayerNorm_type=self.layernorm_type)
        self.msfn = MSFN(in_channels=self.dim)

    def forward(self, x):
        serb = x + self.serb(self.norm1(x))
        msfn = serb + self.msfn(self.norm2(serb))
        return msfn


class CPS(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=16,  num_blocks=[2, 4, 4, 6], num_refinement_blocks=2, bias=False, LayerNorm_type='WithBias'):
        super(CPS, self).__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        self.dim = dim

        self.patch_embed = OverlapPatchEmbed(in_c=self.in_ch, embed_dim=self.dim)

        self.encoder_level_1 = nn.Sequential(*[CPSB(dim=self.dim, LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[0])])
        self.INR1 = INR(self.dim)
        self.down1_2 = Downsample(self.dim)

        self.encoder_level_2 = nn.Sequential(*[CPSB(dim=int(self.dim * 2 ** 1), LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[1])])
        self.INR2 = INR(self.dim * 2 ** 1)
        self.down2_3 = Downsample(int(self.dim * 2 ** 1))

        self.encoder_level_3 = nn.Sequential(*[CPSB(dim=int(self.dim * 2 ** 2), LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[2])])
        self.INR3 = INR(self.dim * 2 ** 2)
        self.down3_4 = Downsample(int(self.dim * 2 ** 2))

        self.latten = nn.Sequential(*[CPSB(dim=int(self.dim * 2 ** 3), LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(self.dim * 2 ** 3))
        self.reduce_channel_3 = nn.Conv2d(int(self.dim * 2 ** 3),
                                          int(self.dim * 2 ** 2),
                                          kernel_size=1,
                                          bias=bias)
        self.decoder_level_3 = nn.Sequential(*[CPSB(dim=int(self.dim * 2 ** 2), LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(self.dim * 2 ** 2))
        self.reduce_channel_2 = nn.Conv2d(int(self.dim * 2 ** 2),
                                          int(self.dim * 2 ** 1),
                                          kernel_size=1,
                                          bias=bias)
        self.decoder_level_2 = nn.Sequential(*[CPSB(dim=int(self.dim * 2 ** 1), LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(self.dim * 2 ** 1))
        self.decoder_level_1 = nn.Sequential(*[CPSB(dim=int(self.dim * 2 ** 1), LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[CPSB(dim=int(self.dim * 2 ** 1), LayerNorm_type=LayerNorm_type, bias=bias) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(self.dim * 2 ** 1), self.out_ch, kernel_size=3, stride=1, padding=1, bias=bias)

        self.RGB2YCbCr = RGBToYCbCrTransform(expansion_factor=96, alpha_R=alpha_r, alpha_G=alpha_g, alpha_B=alpha_b)
        self.YCbCr2RGB_max = YCbCrToRGBTransform(expansion_factor=96, beta_Y=beta_y, beta_Cb=beta_cb, beta_Cr=beta_cr)
        self.YCbCr2RGB_mid = YCbCrToRGBTransform(expansion_factor=96, beta_Y=beta_y, beta_Cb=beta_cb, beta_Cr=beta_cr)
        self.YCbCr2RGB_min = YCbCrToRGBTransform(expansion_factor=96, beta_Y=beta_y, beta_Cb=beta_cb, beta_Cr=beta_cr)

    def forward(self, img):
        YCbCr_images = self.RGB2YCbCr(img)

        Y_images  = YCbCr_images[:, 0, :, :]
        Cb_images = YCbCr_images[:, 1, :, :]
        Cr_images = YCbCr_images[:, 2, :, :]
        Y_images  = Y_images.unsqueeze(1).cuda()
        Cb_images = Cb_images.unsqueeze(1).cuda()
        Cr_images = Cr_images.unsqueeze(1).cuda()

        Cb_images_mid = F.interpolate(Cb_images, scale_factor=0.5)
        Cb_images_min = F.interpolate(Cb_images, scale_factor=0.25)

        Cr_images_mid = F.interpolate(Cr_images, scale_factor=0.5)
        Cr_images_min = F.interpolate(Cr_images, scale_factor=0.25)

        images_mid = F.interpolate(img, scale_factor=0.5)
        images_min = F.interpolate(img, scale_factor=0.25)

        INR_list = []

        in_enc_level1 = self.patch_embed(Y_images)
        out_enc_level1 = self.encoder_level_1(in_enc_level1)
        INR1 = self.INR1(out_enc_level1)
        ycbcr_max = torch.cat([INR1, Cb_images, Cr_images], dim=1)
        INRrgb_max = self.YCbCr2RGB_max(ycbcr_max)
        INR_list.append(INRrgb_max)

        in_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level_2(in_enc_level2)
        INR2 = self.INR2(out_enc_level2)
        ycbcr_mid = torch.cat([INR2, Cb_images_mid, Cr_images_mid], dim=1)
        INRrgb_mid = self.YCbCr2RGB_mid(ycbcr_mid)
        INR_list.append(INRrgb_mid)

        in_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level_3(in_enc_level3)
        INR3 = self.INR3(out_enc_level3)
        ycbcr_min = torch.cat([INR3, Cb_images_min, Cr_images_min], dim=1)
        INRrgb_min = self.YCbCr2RGB_min(ycbcr_min)
        INR_list.append(INRrgb_min)

        in_latten = self.down3_4(out_enc_level3)
        out_latten = self.latten(in_latten)

        in_dec_level3 = self.up4_3(out_latten)
        in_dec_level3 = torch.cat([in_dec_level3, out_enc_level3], dim=1)
        in_dec_level3 = self.reduce_channel_3(in_dec_level3)
        out_dec_level3 = self.decoder_level_3(in_dec_level3)

        in_dec_level2 = self.up3_2(out_dec_level3)
        in_dec_level2 = torch.cat([in_dec_level2, out_enc_level2], dim=1)
        in_dec_level2 = self.reduce_channel_2(in_dec_level2)
        out_dec_level2 = self.decoder_level_2(in_dec_level2)

        in_dec_level1 = self.up2_1(out_dec_level2)
        in_dec_level1 = torch.cat([in_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level_1(in_dec_level1)

        out_ref = self.refinement(out_dec_level1)
        out = self.output(out_ref)
        return out, INR_list



class IGA(nn.Module):
    def __init__(self, dim, num_heads, bias) -> None:
        super(IGA, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.conv_in = nn.Conv2d(in_channels=dim, out_channels=dim*3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(in_channels=dim*3, out_channels=dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=False)

        self.conv_out = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.conv_in(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.conv_out(out)
        return out


class SPC(nn.Module):
    def __init__(self, dim, bias=False):
        super(SPC, self).__init__()
        self.dim = dim
        self.conv1x1 = nn.Sequential(nn.Conv2d(in_channels=self.dim, out_channels=int(self.dim // 4), kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.ReLU(),
                                     )
        self.conv3x3 = nn.Sequential(nn.Conv2d(in_channels=self.dim, out_channels=int(self.dim // 2), kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.Conv2d(in_channels=int(self.dim // 2), out_channels=int(self.dim // 2), kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.ReLU(),
                                     )
        self.conv5x5 = nn.Sequential(nn.Conv2d(in_channels=self.dim, out_channels=int(self.dim // 8), kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.Conv2d(in_channels=int(self.dim // 8), out_channels=int(self.dim // 8), kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.Conv2d(in_channels=int(self.dim // 8), out_channels=int(self.dim // 8), kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.ReLU(),
                                     )
        self.maxpool3x3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                        nn.Conv2d(in_channels=self.dim, out_channels=self.dim // 8, kernel_size=1, stride=1, padding=0, bias=False),
                                        nn.ReLU(),
                                        )
        self.conv_out = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x1 = x.clone()
        x2 = x.clone()
        x3 = x.clone()
        x4 = x.clone()
        x1_ = self.conv1x1(x1)
        x2_ = self.conv3x3(x2)
        x3_ = self.conv5x5(x3)
        x4_ = self.maxpool3x3(x4)
        out = self.conv_out(torch.cat((x1_, x2_, x3_, x4_), dim=1))
        return out


class FPSB(nn.Module):
    def __init__(self, dim, num_heads, LayerNorm_type,  bias=False):
        super(FPSB, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.layernorm_type = LayerNorm_type
        self.norm1 = LayerNorm(dim=self.dim, LayerNorm_type=self.layernorm_type)
        self.iga = IGA(dim=self.dim, num_heads=self.num_heads, bias=bias)
        self.norm2 = LayerNorm(dim=self.dim, LayerNorm_type=self.layernorm_type)
        self.spc = SPC(dim=self.dim, bias=bias)

    def forward(self, x):
        x = x + self.iga(self.norm1(x))
        x = x + self.spc(self.norm2(x))
        return x


class FPS(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=16, num_blocks=[2, 4, 4, 6], num_refinement_blocks=2, heads=[1, 2, 4, 6], bias=False, LayerNorm_type='WithBias') -> None:
        super(FPS, self).__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.dim = dim

        self.patch_embed = OverlapPatchEmbed(in_c=self.in_ch, embed_dim=self.dim)

        self.encoder_level_1 = nn.Sequential(*[FPSB(dim=self.dim, num_heads=heads[0], LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])]) 
        self.down1_2 = Downsample(self.dim)

        self.encoder_level_2 = nn.Sequential(*[FPSB(dim=int(self.dim * 2 ** 1), num_heads=heads[1], LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(self.dim * 2 ** 1))

        self.encoder_level_3 = nn.Sequential(*[FPSB(dim=int(self.dim * 2 ** 2), num_heads=heads[2], LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(self.dim * 2 ** 2))

        self.latten = nn.Sequential(*[FPSB(dim=int(self.dim * 2 ** 3), num_heads=heads[3], LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(self.dim * 2 ** 3))
        self.reduce_channel_3 = nn.Conv2d(int(self.dim * 2 ** 3),
                                          int(self.dim * 2 ** 2),
                                          kernel_size=1,
                                          bias=bias)
        self.decoder_level_3 = nn.Sequential(*[FPSB(dim=int(self.dim * 2 ** 2), num_heads=heads[2], LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(self.dim * 2 ** 2))
        self.reduce_channel_2 = nn.Conv2d(int(self.dim * 2 ** 2),
                                          int(self.dim * 2 ** 1),
                                          kernel_size=1,
                                          bias=bias)
        self.decoder_level_2 = nn.Sequential(*[FPSB(dim=int(self.dim * 2 ** 1), num_heads=heads[1], LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(self.dim * 2 ** 1))
        self.decoder_level_1 = nn.Sequential(*[FPSB(dim=int(self.dim * 2 ** 1), num_heads=heads[0], LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[FPSB(dim=int(self.dim * 2 ** 1), num_heads=heads[0], LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(self.dim * 2 ** 1), self.out_ch, kernel_size=3, stride=1, padding=1, bias=bias)

        self.INR1_expand = nn.Conv2d(in_channels=3, out_channels=int(self.dim * 2 ** 0), kernel_size=1, stride=1, padding=0, bias=bias)
        self.INR2_expand = nn.Conv2d(in_channels=3, out_channels=int(self.dim * 2 ** 1), kernel_size=1, stride=1, padding=0, bias=bias)
        self.INR3_expand = nn.Conv2d(in_channels=3, out_channels=int(self.dim * 2 ** 2), kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, img, INR_list):
        INRrgb_max = INR_list[0]
        INR1 = self.INR1_expand(INRrgb_max)
        
        INRrgb_mid = INR_list[1]
        INR2 = self.INR2_expand(INRrgb_mid)

        INRrgb_min = INR_list[2]
        INR3 = self.INR3_expand(INRrgb_min)

        in_enc_level1 = self.patch_embed(img)
        out_enc_level1 = self.encoder_level_1(in_enc_level1)
        out_enc_level1 = out_enc_level1 + INR1

        in_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level_2(in_enc_level2)
        out_enc_level2 = out_enc_level2 + INR2

        in_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level_3(in_enc_level3)
        out_enc_level3 = out_enc_level3 + INR3

        in_latten = self.down3_4(out_enc_level3)
        out_latten = self.latten(in_latten)

        in_dec_level3 = self.up4_3(out_latten)
        in_dec_level3 = torch.cat([in_dec_level3, out_enc_level3], dim=1)
        in_dec_level3 = self.reduce_channel_3(in_dec_level3)
        out_dec_level3 = self.decoder_level_3(in_dec_level3)

        in_dec_level2 = self.up3_2(out_dec_level3)
        in_dec_level2 = torch.cat([in_dec_level2, out_enc_level2], dim=1)
        in_dec_level2 = self.reduce_channel_2(in_dec_level2)
        out_dec_level2 = self.decoder_level_2(in_dec_level2)

        in_dec_level1 = self.up2_1(out_dec_level2)
        in_dec_level1 = torch.cat([in_dec_level1, out_enc_level1], dim=1)
        out_dec_level1 = self.decoder_level_1(in_dec_level1)

        out_ref = self.refinement(out_dec_level1)
        out = self.output(out_ref)
        return out


class Model_(nn.Module):
    def __init__(self, CPS, FPS) -> None:
        super().__init__()

        self.cps = CPS
        self.fps = FPS
        self.RGB2YCbCr = RGBToYCbCrTransform(expansion_factor=64, alpha_R=alpha_r, alpha_G=alpha_g, alpha_B=alpha_b)
        self.YCbCr2RGB = YCbCrToRGBTransform(expansion_factor=64, beta_Y=beta_y, beta_Cb=beta_cb, beta_Cr=beta_cr)

    def forward(self, img):
        rgb_img = img.clone()
        img_llm = img.clone()

        YCbCr_images = self.RGB2YCbCr(rgb_img)

        Y_hat, INR_list = self.cps(img_llm)
        y_hat = Y_hat.squeeze(1)

        YCbCr_images[:, 0, :, :] = y_hat

        RGB_image = self.YCbCr2RGB(YCbCr_images)

        image = self.fps((RGB_image+img), INR_list)

        return Y_hat, image


CPS_model = CPS(in_channels=1,
                      out_channels=1,
                      dim=32,
                      num_blocks=[1, 2, 2, 4],
                      num_refinement_blocks=2,
                      bias=False,
                      LayerNorm_type='WithBias').cuda()


FPS_model = FPS(in_channels=3, 
                    out_channels=3, 
                    dim=48,
                    num_blocks=[1, 2, 2, 4],
                    num_refinement_blocks=2, 
                    heads=[1, 2, 4, 4],
                    bias=False, 
                    LayerNorm_type='WithBias').cuda()


CSTNet = Model_(CPS_model, FPS_model).cuda()

