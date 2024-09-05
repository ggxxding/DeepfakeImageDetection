import gradio as gr
import torch
import torchvision.transforms as transforms
import numpy as np
from networks.efficientnet import efficientnet_b1
from torchvision.utils import save_image
from torch.nn import functional as F

def interpolate(img, factor):
    return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest', recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)

model_path = "./checkpoints/NPR_effnetb12024_08_21_15_18_02/model_epoch_8.pth"
root = "/mnt/share_data/dmj/phase1_converted/train/"
toTensors = transforms.ToTensor()
toPIL = transforms.ToPILImage()
norm_func = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
crop_func = transforms.CenterCrop(224)
rz_func = transforms.Resize((256, 256))
rz_func2 = transforms.Resize((96, 120))
rz_func3 = transforms.Resize((384, 384))
preprocess = transforms.Compose(
    [
        toTensors,
        rz_func,
        crop_func,
        norm_func,
    ]
)

def deepfake_image_detection(image):
    image_tensor = toTensors(image)
    NPR_ = interpolate(image_tensor, 0.5)
    NPR = image_tensor - NPR_
    NPR = rz_func3(toPIL(NPR))
    NPR_ = toPIL(NPR_)
    image = preprocess(image).unsqueeze(0).cuda()
    output, flat, NPR1_ = model(image)
    output = output.sigmoid().flatten().tolist()[0]
    flat = rz_func2(toPIL(flat))

    return [NPR, flat, {"Real": 1-output, "Fake": output }]

if __name__ == "__main__":
    model = efficientnet_b1(num_classes = 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model.cuda()
    model.eval()

    # imgs = ['/mnt/share_data/dmj/phase1_converted/train/0_real/d80534da7b09be2684de429be22b8b9d.jpg',
    #         '/mnt/share_data/dmj/phase1_converted/train/0_real/f7ade355ac7d53dfdd39b2d71d72bfa2.jpg',
    #         '/mnt/share_data/dmj/phase1_converted/train/1_fake/b1e9f64896e0b11945a9946258f7218d.jpg',
    #         '/mnt/share_data/dmj/phase1_converted/train/1_fake/6ad80d61f821a5e58878d58a2018eb48.jpg']
    imgs = ['./src/d80534da7b09be2684de429be22b8b9d.jpg',
            './src/f7ade355ac7d53dfdd39b2d71d72bfa2.jpg',
            './src/b1e9f64896e0b11945a9946258f7218d.jpg',
            './src/6ad80d61f821a5e58878d58a2018eb48.jpg']
    demo = gr.Interface(
        fn=deepfake_image_detection,
        inputs=[gr.Image(type='pil')],
        outputs=[gr.Image(label = 'Neighboring pixel relationships',type='pil'),
                 gr.Image(label = 'Lase hidden layer',type='pil'),
                 gr.Label(num_top_classes=2)],
        examples = imgs
    )

    demo.launch()