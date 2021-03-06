import numpy as np
from PIL import Image
from CLIP import clip # The clip model
from torchvision import transforms # Some image transforms
import gc
from io import BytesIO
from preprocess import process_transformers
import sys
sys.path.append('./taming-transformers')
from source.utils import *
from source.textmodel import *
import streamlit as st

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

# The transforms to get variations of image
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
        z = rand_z(model, int(width), int(height), CFG.device)
    z.requires_grad=True

    # The text target
    with torch.no_grad():
        text_embed = text_model.encode_text([process_transformers(prompt_text)]).to(device).float()
        neg_text_embed = text_model.encode_text([process_transformers(negative_prompt)]).to(device).float()

    # The optimizer - feel free to try different ones here
    optimizer = torch.optim.Adam([z], lr=lr, weight_decay=1e-6)

    losses = [] # Keep track of our losses (RMSE values)

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
        output = synth(model, z)

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
    