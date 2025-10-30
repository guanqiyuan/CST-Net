

 <div align=center> <img src="https://github.com/guanqiyuan/CST-Net/blob/main/figs/CST-Net.png" width="210px"> </div>
 
 #  <center>  Rethinking Nighttime Image Deraining via Learnable Color Space Transformation  (NeurIPS 2025)

> [Qiyuan Guan](https://guanqiyuan.github.io/)* <sup>1</sup>, [Xiang Chen](https://cschenxiang.github.io/)* <sup>2</sup>, Guiyue Jin <sup>1</sup>, Jiyu Jin <sup>1</sup>, [Shumin Fan](https://scholar.google.com/citations?user=WZv2NgoAAAAJ&hl=zh-CN&authuser=1) <sup>3</sup>, [Tianyu Song](https://scholar.google.com/citations?user=wA3Op6cAAAAJ&hl=zh-CN) <sup>3</sup>, [Jinshan Pan](https://jspan.github.io/) <sup>2</sup>
>
> Dalian Polytechnic University<sup>1</sup>, Nanjing University of Science and Technology<sup>2</sup>, Dalian Martime University<sup>3</sup>


> [[Paper](https://arxiv.org/abs/2510.17440)]


> 
 **👉️ Welcome to visit our website (专注底层视觉领域的信息服务平台) for low-level vision: [https://lowlevelcv.com/](https://lowlevelcv.com/)**

---

## ⛳️ To do

* ✅ Release the code
* ✅ Release the visual results
* ✅ Release the dataset


<!--
✅
❎
-->

---

### ⬇️ HQ-NightRain Dataset Download
| Download Link |
|---------|
| [Google Drive](https://drive.google.com/drive/folders/1WZ3uIWkNkIHVLGDUo8OPmrsNI_hheIUT?usp=sharing) / [Baidu Netdisk](https://pan.baidu.com/s/1gU2aI08UFzS3JLSl_l0GLw) (asht) |

---


<!--
## 📘 Quantitative Results
![image](url)

---
-->



## 🛠 Setup
* Type the command:
```
conda env create -f environment.yml
conda activate CSTNet
```
* If torch1.10 download fails, please run the following command：
```
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```


## 📷️ Visual Results

#### ● Comparative method results
>
> 
| Datasets | Visual Results Download Link |
|---------|------|
| HQ-NightRain-RS | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1zmLlXX09j5fIjHbil2Uy-g) (adai) |
| HQ-NightRain-RD | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1L6hlOwfwZ-J8gTg9H3XsMA) (m1kb) |
| HQ-NightRain-SD | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1ZTxNP2WPsH7J1CYQ3wy5Aw) (k7aq) |
| GTAV-NightRain | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/12n9t8MytFJMu5MJyGcP1iA) (b6sm) |
| RealRain1k-L | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1rnoXBv8KwqgYHq7G_GWFhw) (zi73) |
| RealRain1k-H | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1WgP6Kh3XgHdmvu4GOp4LIg) (r4cr) |
| RainDS-real-RS | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1Rth8lMC4TmN-QNeY9sqpDg) (98bt) |
| RainDS-real-RD | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/14OKpfrO1uaOejUtVU1YJtA) (1ed3) |
| RainDS-real-RDS | Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1RPMb104it5M2L_GeJ-yGEA) (9mhh) |


#### ● CST-Net results
| Visual Results Download Link |
|------|
|Google Drive / [Baidu Netdisk](https://pan.baidu.com/s/1ADwKhb720BOS2g-UpFmKJw) (n3vg)|

Due to storage limitations, please contact us to obtain the Google Drive link.

---


<!--
## 📊 Pre-trained Models
>
| Datasets | Visual Results Download Link |
|---------|------|
| HQ-NightRain-RS | Google Drive / [Baidu Netdisk]() () |
| HQ-NightRain-RD | Google Drive / [Baidu Netdisk]() () |
| HQ-NightRain-SD | Google Drive / [Baidu Netdisk]() () |
| GTAV-NightRain | Google Drive / [Baidu Netdisk]() () |
| RealRain1k | Google Drive / [Baidu Netdisk]() () |
| RainDS-real | Google Drive / [Baidu Netdisk]() () |
-->


## 🧮 Evaluation

#### ● Run the following code to obtain the output visual results

```
python test.py
```
And you can find the output visual results in the folder " results/test/  ".

#### ● Install the environment
We use the code provided by [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) for evaluation. Thanks to Chaofeng Chen!
```
pip install pyiqa
```
#### ● Run the following command to calculate the metrics

```
python cal_metrics.py --inp_imgs ./results --gt_imgs ./dataset/test/target --log path_save_log
```


## 💪 Training
Run the following code to start training.
```
python train.py
```


## 👍 Acknowledgement

Thanks for their awesome works ([IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) and [NeRD-Rain](https://github.com/cschenxiang/NeRD-Rain?tab=readme-ov-file))

---

## ❣ Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{guan2025cstnet,
  title={Rethinking Nighttime Image Deraining via Learnable Color Space Transformation},
  author={Guan, Qiyuan and Chen, Xiang and Jin, Guiyue and Jin, Jiyu and Fan, Shumin and Song, Tianyu and Pan, Jinshan},
  journal={NeurIPS},
  year={2025}
}
```


## 📧 Contact
If you have any questions, please feel free to contact qyuanguan@gmail.com.


---
<a href="https://info.flagcounter.com/6y1p"><img src="https://s01.flagcounter.com/count2/6y1p/bg_FFFFFF/txt_000000/border_CCCCCC/columns_2/maxflags_10/viewers_0/labels_1/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>
