from ast import Bytes
from cgitb import text
import torch 
from torch import nn
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import IPython.display as ipd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.models import vgg16
from CLIP import clip # The clip model
from torchvision import transforms # Some useful image transforms
import torch.nn.functional as F # Some extra methods we might need
from omegaconf import OmegaConf
import sys
sys.path.append('./taming-transformers')
from taming.models import cond_transformer, vqgan
import os
import cv2
import gc
import pandas as pd
import itertools
import albumentations as A
import matplotlib.pyplot as plt
import timm
import streamlit as st
from io import BytesIO
from transformers import (
    CamembertModel,
    CamembertTokenizer,
    CamembertConfig,
)
from preprocess import process_transformers
from urllib.request import urlopen
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

#@title Helper function
def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward
 
    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)
 
 
replace_grad = ReplaceGrad.apply
 
 
class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)
 
    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

 
clamp_with_grad = ClampWithGrad.apply

def vector_quantize(x, codebook):
  d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
  indices = d.argmin(-1)
  x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
  return replace_grad(x_q, x)

def synth(z):
  z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
  return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)

def rand_z(width, height):
  f = 2**(model.decoder.num_resolutions - 1)
  toksX, toksY = width // f, height // f
  n_toks = model.quantize.n_e
  one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
  z = one_hot @ model.quantize.embedding.weight
  z = z.view([-1, toksY, toksX, model.quantize.e_dim]).permute(0, 3, 1, 2)
  return z

def clip_loss(im_embed, text_embed):
  im_normed = F.normalize(im_embed.unsqueeze(1), dim=2)
  text_normed = F.normalize(text_embed.unsqueeze(0), dim=2)
  dists = im_normed.sub(text_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2) # Squared Great Circle Distance
  return dists.mean()

def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m

class CFG:
    # captions_path = captions_path
    batch_size = 32
    num_workers = 2
    head_lr = 1e-3
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder_model = "wangchanberta-base-att-spm-uncased"
    text_embedding = 768
    text_tokenizer = "wangchanberta-base-att-spm-uncased"
    max_length = 200

    pretrained = True
    trainable = True
    temperature = 1.0

    num_projection_layers = 1
    projection_dim = 512 
    dropout = 0.1

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = CamembertModel.from_pretrained(model_name)
        else:
            self.model = CamembertModel(config=CamembertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class TextModel(nn.Module):
    def __init__(
        self,
        text_embedding = CFG.text_embedding
    ):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.tokenizer = CamembertTokenizer.from_pretrained(CFG.text_tokenizer)

    def forward(self, batch):
        # Getting Text Features
        text_features = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        # Project to the same dim of image encoder
        text_embeddings = self.text_projection(text_features)

        return text_embeddings
    
    def encode_text(self, text):
        tokened_word = self.tokenizer(text, padding=True, truncation=True, max_length=CFG.max_length)
        text_features = self.text_encoder(
            input_ids=torch.tensor(tokened_word["input_ids"]).to(CFG.device),
            attention_mask=torch.tensor(tokened_word["attention_mask"]).to(CFG.device)
        )
        text_embeddings = self.text_projection(text_features)
        return text_embeddings

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                # A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
def add_to_prompt(text):
    global prompt_text
    st.session_state.prompt_text = prompt_text + " " + text

input_help = "ถ้าเว้นวรรคแล้วใส่คำว่า \"ภาพสวย\" ต่อท้ายจะทำให้ภาพสวยขึ้น!"
neg_help = "โมเดลจะพยายามทำให้สิ่งเหล่านี้อยู่ในภาพน้อยที่สุด"

st.write("# VQGANxThCLIP -- สร้างรูปภาพจากข้อความ")
prompt_text = st.text_input("ใส่คำเพื่อสร้างรูป", key="user_input", help=input_help)
with st.expander("เพิ่มสไตล์ของภาพ"):
    negative_prompt = st.text_input("เพิ่มสิ่งที่ไม่อยากให้อยู่ในภาพ", value='ภาพเบลอ', help=neg_help)
    st.write("เพิ่มสไตล์ของภาพโดยใส่คำเหล่านี้ (สามารถใส่มากกว่า 1 สไตล์ได้!)")
    col1, col2, col3, col4 =  st.columns(4)
    listofenhancers = [
        "ภาพสวย",
        "ภาพยนตร์",
        "ภาพสีอะคริลิค",
        "ภาพสีน้ำ",
        "ภาพการ์ตูน",
        "ภาพคุณภาพสูง",
        "ภาพวาด",
        "ภาพ 3 มิติ",
        "ภาพวาดโดยเด็ก",
        "ภาพแฟนตาซี",
        "ภาพประติมากรรม",
        "1990",
        "ภาพสีน้ำมัน",
        "ภาพวาดด้วยดินสอ",
        "ภาพเกม",
    ]
    for i, enhancer in enumerate(listofenhancers):
        if i%4 == 0:
            with col1:
                st.button(enhancer, on_click=add_to_prompt, args=(enhancer,))
        elif i%4 == 1:
            with col2:
                st.button(enhancer, on_click=add_to_prompt, args=(enhancer,))
        elif i%4 == 2:
            with col3:
                st.button(enhancer, on_click=add_to_prompt, args=(enhancer,))
        elif i%4 == 3:
            with col4:
                st.button(enhancer, on_click=add_to_prompt, args=(enhancer,))

with st.expander("ตั้งค่าโมเดล"):
    col1, col2, col3 = st.columns(3)

    with col1:
        iters = st.number_input('Number of steps', value=300, min_value=10, step=10)
    with col2:
        width = st.number_input("Width", value=256, min_value=64, step=64)
    with col3:
        height = st.number_input("Height", value=256, min_value=64, step=64)
    
    init_image = st.file_uploader("ใส่รูปเริ่มต้น (optional)")
    target_image = st.file_uploader("ใส่รูปเพื่อเป็นเป้าหมาย (optional)")
    
    st.write("Advanced setting (optional)")
    scol1, scol2, scol3 = st.columns(3)

    with scol1:
        neg_weight = st.number_input('Negative weight', value=0.4, step=0.05)
        lr = st.number_input('Learning rate', value=0.1, min_value=0., max_value=1.0, step=0.05)
    with scol2:
        target_weight = st.number_input('Target image weight', value=0., min_value=0., max_value=1.0, step=0.05)
        crops_per_iteration = st.number_input('Crops per iteration', value=8, min_value=1, step=1)
    with scol3:
        aesthetic_weight = st.number_input('Aesthetic weight', value=0.005, step=0.0005, format=f'%.4f')

submit = st.button("Let's generate!")
last_step = st.empty()
bar = st.empty()
image_holder = st.empty()
download = st.empty()

# The transforms to get variations of our image
tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomAdjustSharpness(1), # game changer
    # transforms.RandomAutocontrast(),
    # transforms.RandomEqualize(),
    transforms.RandomAffine(5),
    transforms.ColorJitter(),
    transforms.GaussianBlur(3),
])


def run():
    global clip_model, device, model, text_model, amodel
    latest_step = st.empty()
    bar = st.empty().progress(0)
    latest_step.text('Loading CLIP Model...')

    clip_model, compose = clip.load('ViT-B/32')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    bar.progress(25)
    latest_step.text('loading VQGAN Model...')
    model = load_vqgan_model('vqgan_imagenet_f16_16384.yaml', 'vqgan_imagenet_f16_16384.ckpt').to(device)

    bar.progress(50)
    latest_step.text('loading Text Model...')
    text_model = TextModel().to(CFG.device)
    text_model.load_state_dict(torch.load("CLIP-MSE-WangchanBerta/text_MSE_2m.pt", map_location=CFG.device))
    text_model.eval().requires_grad_(False)

    bar.progress(75)
    latest_step.text('loading Aesthetic Model...')
    amodel= get_aesthetic_model(clip_model="vit_b_32").to(CFG.device)
    amodel.eval()

    bar.progress(100)
    latest_step.text('Finishing up...')
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    latest_step.empty()
    bar.empty()

    if target_image is not None:
        if 'http' in target_image:
            img = Image.open(urlopen(target_image)).convert('RGB').resize((224, 224))
        else:
            img = Image.open(target_image).convert('RGB').resize((224, 224))
            im = torch.tensor(np.array(img)).permute(2, 0, 1)/255
            im = im.unsqueeze(0).to(device)
        with torch.no_grad():
            tar_embed = clip_model.encode_image(normalize(im).to(device)).float()

    # The z we'll be optimizing
    if init_image is not None:
        if 'http' in init_image:
            img = Image.open(urlopen(init_image)).convert('RGB').resize((width, height))
        else:
            img = Image.open(init_image).convert('RGB').resize((width, height))
            im = torch.tensor(np.array(img)).permute(2, 0, 1)/255
            im = im.unsqueeze(0).to(device)
            z, *_ = model.encode(im)
    else:
        z = rand_z(int(width), int(height))
    z.requires_grad=True

    # The text target
    with torch.no_grad():
        text_embed = text_model.encode_text([process_transformers(prompt_text)]).to(device).float()
        neg_text_embed = text_model.encode_text([process_transformers(negative_prompt)]).to(device).float()

    # The optimizer - feel free to try different ones here
    optimizer = torch.optim.Adam([z], lr=lr, weight_decay=1e-6)

    losses = [] # Keep track of our losses (RMSE values)

    # A folder to save results
    # !rm -r steps
    # !mkdir steps


    # Display for showing progress
    # fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # p = display(fig, display_id=True)

    # The optimization loop:
    for i in range(int(iters)):
        last_step.write(f'{i+1} / {int(iters)}')
        bar.progress(int((i+1)/iters * 100))
    # Reset everything related to gradient calculations
        optimizer.zero_grad()

        # Get the GAN output
        output = synth(z)

        # Calculate our loss across several different random crops/transforms
        loss = 0
        for _ in range(int(crops_per_iteration)):
            image_embed = clip_model.encode_image(normalize(tfms(output)).to(device)).float()
            # text-image loss
            loss += clip_loss(image_embed, text_embed)/crops_per_iteration
            # text-image neg loss
            loss -= (neg_weight * clip_loss(image_embed, neg_text_embed))/crops_per_iteration
            # aesthetic weight loss
            loss -= (aesthetic_weight * amodel(image_embed/image_embed.norm(dim=-1, keepdim=True))[0, 0])/crops_per_iteration
            # target image loss
            if target_image:
                loss += (target_weight * clip_loss(image_embed, tar_embed))/crops_per_iteration

        # Store loss
        losses.append(loss.detach().item())  
        # Save image
        im_arr = np.array(output.cpu().squeeze().detach().permute(1, 2, 0)*255).astype(np.uint8)
        # Image.fromarray(im_arr).save(f'steps/{i:04}.jpeg')
        Image.fromarray(im_arr).save(f'steps/{prompt_text}.jpeg')

        # Update plots 
        if i % 5 == 0: # Saving time
            im_arr = np.array(output.cpu().squeeze().detach().permute(1, 2, 0)*255).astype(np.uint8)
            image_holder.image(Image.fromarray(im_arr))

        # Backpropagate the loss and use it to update the parameters
        loss.backward() # This does all the gradient calculations
        optimizer.step() # The optimizer does the update
    
    last_step.empty()
    bar.empty()

    # ipd.clear_output()

if submit:
    download.empty()
    run()
    gc.collect()
    torch.cuda.empty_cache()
    image = Image.open(f'steps/{prompt_text}.jpeg')
    buf = BytesIO()
    image.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    download.download_button(
        label="Download picture",
        data=byte_im,
        file_name=prompt_text+'.jpeg',
        mime="image/jpeg"
    )
    