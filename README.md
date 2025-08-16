# Awesome-Training-Free-WAN2.1-Editing🍀  
**Star🌟 is a great help in open source development!**
  
*Awesome Training-Free (Inversion-Free) methods meet WAN2.1-T2V.*  
- WAN2.1 + FlowEdit
- WAN2.1 + FlowAlign
- WANAlign2.1⚡ (FlowAlign + masking method)

# Introduce WANAlign2.1⚡
<p align="center">
  <img src="./utils/model.gif" alt="animated"/>
</p>  
<p align="center">
    <a href="https://kyujinpy.tistory.com/178"><img alt="Static Badge" src="https://img.shields.io/badge/Blog-WANAlign2.1-orange?label=Blog"></a>
    <a href="https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing/tree/master/diffusers"><img alt="Static Badge" src="https://img.shields.io/badge/Diffusers-yellow?label=Library"></a>
</p>

We present **WANAlign2.1⚡**, an inversion-free video editing framework that combines the inversion-free editing method **FlowAlign** with **WAN2.1**. By integrating FlowAlign’s inversion-free sampling equation into WAN2.1, our approach preserves the intrinsic characteristics of the source video during editing.   
To further enhance control, we introduce **Decoupled Inversion-Free Sampling (DIFS)**, which leverages **attention masking to independently adjust the editing strength** between preserved and modified regions.  
The previous methods frequently modified regions that should have been preserved, thereby degrading overall consistency. Our **WANAlign2.1⚡** achieves improved spatial-temporal consistency and enhanced text-guided editing performance through DIFS.   
As shown in the [Results](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing?tab=readme-ov-file#results), qualitative results demonstrate that our method is state-of-the-art.

# Results🐦‍🔥
### 0️⃣Inference speed
| WANAlign2.1⚡| FlowDirector | WANEdit2.1 |
| --- | --- | --- |
| **75 seconds** | 540 seconds | 150 seconds |
> A100 80GB GPU

### 1️⃣Color/Background Editing

<table border="0" width="100%">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>WANAlign2.1⚡</b></td>
  <td style="text-align:center;"><b>FlowDirector</b></td>
  <td style="text-align:center;"><b>WANEdit2.1</b></td>
</tr>
<tr>
  <td><img src="https://github.com/user-attachments/assets/5153cf99-9042-4dde-8b38-4b8988b96301"></td>
  <td><img src="https://github.com/user-attachments/assets/6c4947c5-efa7-48d5-815e-5b582952c690"></td>
  <td><img src="https://github.com/user-attachments/assets/1ec1c79c-5928-4b91-a5be-bc27e3e7c671"></td>              
  <td><img src="https://github.com/user-attachments/assets/ddde08a1-390d-4963-bdaf-d911c2eed2b0"></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">A large brown bear ... ➡️ A large <b>yellow</b> bear ...</td>
</tr>
    
<tr>
  <td><img src="https://github.com/user-attachments/assets/edc2d5bf-1347-4ac5-86d2-ccdd0a4a1f3e"></td>
  <td><img src="https://github.com/user-attachments/assets/72235e9a-24f9-46a6-a172-125c715b6767"></td>
  <td><img src="https://github.com/user-attachments/assets/76a67bca-0216-444b-b23b-b263d3393c44"></td>              
  <td><img src="https://github.com/user-attachments/assets/f4f15adf-c0e5-44b5-a8d7-9313cb135dd0"></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">... in a snowy field. ➡️ ... in the <b>ocean</b>.</td>
</tr>
</table>

### 2️⃣Object Editing

<table border="0" width="100%">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>WANAlign2.1⚡</b></td>
  <td style="text-align:center;"><b>FlowDirector</b></td>
  <td style="text-align:center;"><b>WANEdit2.1</b></td>
</tr>
<tr>
  <td><img src="https://github.com/user-attachments/assets/6f8231cf-5ac9-499c-a6bb-38f5c39798a8"></td>
  <td><img src="https://github.com/user-attachments/assets/c31bb549-389e-4d54-91dc-fe8ac3ed57f9"></td>
  <td><img src="https://github.com/user-attachments/assets/de23489a-5b5e-4938-92e0-1248664d4a9f"></td>              
  <td><img src="https://github.com/user-attachments/assets/a524ffba-f92f-480c-918b-fd0c3d7acbdd"></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">A graceful sea turtle ... ➡️ A graceful <b>seal</b> ...</td>
</tr>
    
<tr>
  <td><img src="https://github.com/user-attachments/assets/5153cf99-9042-4dde-8b38-4b8988b96301"></td>
  <td><img src="https://github.com/user-attachments/assets/e1a6471e-5438-4743-a5e7-9bf96a7e5c32"></td>
  <td><img src="https://github.com/user-attachments/assets/c293bf71-9609-4798-88e3-0e838b8af96c"></td>              
  <td><img src="https://github.com/user-attachments/assets/5b6b1773-11fa-4f95-9396-dcc42c59f04f"></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">A large brown bear ... ➡️ A large <b>tiger</b> ...</td>
</tr>
</table>

### 3️⃣Texture Editing

<table border="0" width="100%">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>WANAlign2.1⚡</b></td>
  <td style="text-align:center;"><b>FlowDirector</b></td>
  <td style="text-align:center;"><b>WANEdit2.1</b></td>
</tr>
<tr>
  <td><img src="https://github.com/user-attachments/assets/ba576a66-4d2f-4a76-91df-636834318b50"></td>
  <td><img src="https://github.com/user-attachments/assets/6fe5dac3-b6f0-4688-b9a4-429828ca2bdf"></td>
  <td><img src="https://github.com/user-attachments/assets/30c3080d-2c26-41fa-ab13-f6babc097158"></td>              
  <td><img src="https://github.com/user-attachments/assets/96748619-8faf-47ba-9bf9-4ead215f5e43"></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">A black swan ... ➡️ A <b>silver statue</b> swan <b>carrying a turtle</b> ...</td>
</tr>
    
<tr>
  <td><img src="https://github.com/user-attachments/assets/6f93744f-3e1a-4cf0-bde2-bc217a4185f9"></td>
  <td><img src="https://github.com/user-attachments/assets/79b98bd2-326a-4629-876a-0a5291370e3d"></td>
  <td><img src="https://github.com/user-attachments/assets/71633710-d7ec-421d-ac92-7c97c3500c4a"></td>              
  <td><img src="https://github.com/user-attachments/assets/e2dbf575-615f-4d4d-ab86-9b66c5e32713"></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">A rabbit ... ➡️ A <b>crochet</b> rabbit ...</td>
</tr>
</table>

### 4️⃣Add Effect/Object

<table border="0" width="100%">
<tr>
  <td style="text-align:center;"><b>Input Video</b></td>
  <td style="text-align:center;"><b>WANAlign2.1⚡</b></td>
  <td style="text-align:center;"><b>FlowDirector</b></td>
  <td style="text-align:center;"><b>WANEdit2.1</b></td>
</tr>
<tr>
  <td><img src="https://github.com/user-attachments/assets/c1945530-3e72-401d-8a30-dac43db9ed47"></td>
  <td><img src="https://github.com/user-attachments/assets/a7991495-2394-4c96-ba54-f1f84157347e"></td>
  <td><img src="https://github.com/user-attachments/assets/9d3d0c6f-c804-4b86-82c9-0fa6fe3ab88e"></td>              
  <td><img src="https://github.com/user-attachments/assets/90a4d4ea-495a-4dc6-86b4-a21bbd42e20e"></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">A white boat ... ➡️ A white boat <b>on fire</b> ...</td>
</tr>
    
<tr>
  <td><img src="https://github.com/user-attachments/assets/c1945530-3e72-401d-8a30-dac43db9ed47"></td>
  <td><img src="https://github.com/user-attachments/assets/75df3ff2-c28f-40bc-9b6c-3ce72cd1da77"></td>
  <td><img src="https://github.com/user-attachments/assets/ffc426e7-3817-48fe-a602-43e98bfb3e6a"></td>              
  <td><img src="https://github.com/user-attachments/assets/2f3a5ae2-669c-4129-b3a9-b33e6e0e438b"></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">... ➡️ ... <b>In the sky, a large eagle is flying forward.</b></td>
</tr>

<tr>
  <td><img src="https://github.com/user-attachments/assets/5153cf99-9042-4dde-8b38-4b8988b96301"></td>
  <td><img src="https://github.com/user-attachments/assets/560c8526-e48e-4a3d-962d-0dc3d10ab299"></td>
  <td><img src="https://github.com/user-attachments/assets/703717ff-2003-4cdb-a975-2d57ea9d10f3"></td>              
  <td><img src="https://github.com/user-attachments/assets/cf24ef9f-2359-4116-b95f-6ae9c59d79ca"></td>
</tr>
<tr>
  <td width=100% style="text-align:center;" colspan="4">A large brown bear ... ➡️ A large brown bear <b>wearing a hat</b> ...</td>
</tr>
</table>

# Quick Start (Code)🥏
## Environment
```bash
git clone https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing.git
cd ./Awesome-Training-Free-WAN2.1-Editing

conda create -n wanalign python=3.10 -y
conda activate wanalign
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install matplotlib omegaconf imageio
pip install transformers==4.51.3 accelerate
pip install imageio[ffmpeg] ftfy
```
  
You **must** install provided `diffusers` library.
```bash
cd ./diffusers
pip install -e .
```
> I used one A100 80GB GPU.
  
## Run code🏂
```python
python awesome_wan_editing.py --config=[__config_yaml_path__]
# python awesome_wan_editing.py --config=./config/object_editing/bear_tiger.yaml
```
> There are some config file [examples](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing/tree/master/config).
  
In **FlowAlign**, there is `zeta_scale`.  
If the value is high, it will be similar to the source video.  
The `bg_zeta_scale` value is only activated when `flag_attnmask` is `True`.  

## Detail Code lines🏫
- FlowEdit Code: [WanPipeline.flowedit](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing/blob/d93d928a88b2f85b1e9d4494dd36182e9459f391/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py#L817)
- FlowAlign Code: [WanPipeline.flowalign](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing/blob/d93d928a88b2f85b1e9d4494dd36182e9459f391/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py#L1204)
- WANAlign2.1⚡ **Core** Code: [WanPipeline.flowalign](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing/blob/d93d928a88b2f85b1e9d4494dd36182e9459f391/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py#L1553)
- Attention Extract Code: [wan_attention](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing/blob/d93d928a88b2f85b1e9d4494dd36182e9459f391/utils/wan_attention.py#L447)
> If you want to visualize attention masking maps, please activate [these code](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing/blob/d93d928a88b2f85b1e9d4494dd36182e9459f391/diffusers/src/diffusers/pipelines/wan/pipeline_wan.py#L1515).

# TODO-list
- [x] Integrating Diffusers🤗
- [x] Code Release
- [x] Support WAN2.1-T2V-1.3B

# References
- [WAN2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)
- [MasaCtrl](https://github.com/TencentARC/MasaCtrl)
- [FlowEdit](https://matankleiner.github.io/flowedit/)
- [FlowAlign](https://arxiv.org/abs/2505.23145)
- [FlowDirector](https://arxiv.org/abs/2506.05046)
