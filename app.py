import gradio as gr
import torch
import torchvision.transforms as transforms
# import os
from networks.efficientnet import efficientnet_b1
from torchvision.utils import save_image
# import random

model_path = "./checkpoints/NPR_effnetb12024_08_21_15_18_02/model_epoch_8.pth"
root = "/mnt/share_data/dmj/phase1_converted/train/"
toTensors = transforms.ToTensor()
norm_func = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
crop_func = transforms.CenterCrop(224)
rz_func = transforms.Resize((256, 256))
preprocess = transforms.Compose(
    [
        toTensors,
        rz_func,
        crop_func,
        norm_func,
    ]
)

def deepfake_image_detection(image):
    image = preprocess(image).unsqueeze(0).cuda()
    # save_image(image,"test.jpg")
    output =  model(image).sigmoid().flatten().tolist()[0]
    return {"Real": 1-output, "Fake": output }

if __name__ == "__main__":
    model = efficientnet_b1(num_classes = 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model.cuda()
    model.eval()

    # imgs0 = random.sample(os.listdir(root + "0_real/"), 2)
    # imgs0 = [root + "0_real/" + x for x in imgs0]
    # imgs1 = random.sample(os.listdir(root + "1_fake/"), 2)
    # imgs1 = [root + "1_fake/" + x for x in imgs1]
    # imgs = imgs0 + imgs1
    # print(imgs)

    imgs = ['/mnt/share_data/dmj/phase1_converted/train/0_real/d80534da7b09be2684de429be22b8b9d.jpg',
            '/mnt/share_data/dmj/phase1_converted/train/0_real/f7ade355ac7d53dfdd39b2d71d72bfa2.jpg',
            '/mnt/share_data/dmj/phase1_converted/train/1_fake/b1e9f64896e0b11945a9946258f7218d.jpg',
            '/mnt/share_data/dmj/phase1_converted/train/1_fake/6ad80d61f821a5e58878d58a2018eb48.jpg']

    demo = gr.Interface(
        fn=deepfake_image_detection,
        inputs=[gr.Image(type='pil')],
        outputs=[gr.Label(num_top_classes=2)],
        examples = imgs
    )

    demo.launch()