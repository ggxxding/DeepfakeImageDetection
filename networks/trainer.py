import functools
import torch
import torch.nn as nn
# from networks.resnet import resnet50,resnet50full
from networks.efficientnet import efficientnet_b1
from networks.base_model import BaseModel, init_weights
from torch.nn import functional as F
# from torchvision.models import resnext50_32x4d,resnet50
# from torchvision.models import efficientnet_b1, vit_b_32,resnet50

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            # self.model = resnet50(pretrained=False, num_classes=1)
            self.model = efficientnet_b1(pretrained=False, num_classes=1)
            # self.model = resnet50full(pretrained=False, num_classes=1)
            # self.model = vit_b_32(pretrained= False, num_classes = 1)

        if not self.isTrain or opt.continue_train:
            # self.model = resnet50(num_classes=1)
            self.model = efficientnet_b1(num_classes=1)
            # self.model = resnet50full(num_classes=1)
            # self.model = vit_b_32(num_classes = 1)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])
 
    def _interpolate(self,img, factor = 0.5):
            return F.interpolate(
                F.interpolate(
                    img, 
                    scale_factor=factor, mode='nearest', 
                    recompute_scale_factor=True), scale_factor=1/factor, mode='nearest', recompute_scale_factor=True)
    
    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.9
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.9} to {param_group["lr"]}')
        print('*'*25)
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        # [batch,3,224,224] -> [batch,1]
        # print(self.input.shape)
        # torchvision.utils.save_image(self.input[0],'/home/ubuntu/dmj/DeepfakeImageDetection/origin.png')
        # NPR  = self.input - self._interpolate(self.input, 0.5)
        # torchvision.utils.save_image(NPR[0],'/home/ubuntu/dmj/DeepfakeImageDetection/NPR.png')
        self.output = self.model(self.input)
        # print(self.output)
        # for name, param in self.model.named_parameters():
        #     if name == "features.1.0.block.0.0.weight":
        #         print(f'{name}: {param.data}')
        #         quit()

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

