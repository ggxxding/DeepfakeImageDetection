import os
import PIL.Image
from patch_generator import smash_n_reconstruct
import cv2
import numpy as np
# convert origin image(512,512) to rich(:256,256) and poor(256:,256) texture patchs (512,256)
# dir = '/mnt/share_data/dmj/phase1_converted/'
# target = '/mnt/share_data/dmj/phase1_converted_patchcraft/'

def image_to_patch(dir, target):
    images = os.listdir(dir)
    print(len(images))
    for i,img in enumerate(images):
        img_path = os.path.join(dir, img)
        img_target_path = os.path.join(target, img)
        r, p = smash_n_reconstruct(img_path)
        r_p = np.concatenate([r,p], axis = 0)
        r_p = PIL.Image.fromarray(r_p)
        r_p.save(img_target_path)
        print(i)


if __name__ == "__main__":
    image_to_patch(dir = '/mnt/share_data/dmj/phase1_converted/train/0_real', 
                   target = '/mnt/share_data/dmj/phase1_converted_patchcraft/train/0_real')
    image_to_patch(dir = '/mnt/share_data/dmj/phase1_converted/train/1_fake', 
                   target = '/mnt/share_data/dmj/phase1_converted_patchcraft/train/1_fake')
    
    image_to_patch(dir = '/mnt/share_data/dmj/phase1_converted/val/0_real', 
                   target = '/mnt/share_data/dmj/phase1_converted_patchcraft/val/0_real')
    image_to_patch(dir = '/mnt/share_data/dmj/phase1_converted/val/1_fake', 
                   target = '/mnt/share_data/dmj/phase1_converted_patchcraft/val/1_fake')