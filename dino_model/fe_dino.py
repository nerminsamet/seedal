import torch
import os

from torch import nn
from torchvision import transforms as pth_transforms

# from utils import load_image
import dino_model.utils as utils
import dino_model.vision_transformer as vits
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DinoModel(nn.Module):
    def __init__(self, feat_type):
        super(DinoModel, self).__init__()

        self.transform = pth_transforms.Compose([
            pth_transforms.Resize((224, 224), interpolation=3),
            # pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        self.model = self.load_model()
        self.feat_type = feat_type


    def load_model(self):
        # ============ building network ... ============
        p = 8
        model = vits.__dict__["vit_base"](patch_size=p, num_classes=0)
        print(f"Model {'vit_base'} {p}x{p} built.")
        model.cuda()
        utils.load_pretrained_weights(model, "", "", "vit_base", p)
        model.eval()
        return model

    def forward(self, images, feat_type='cls'):
        """Extract the image feature vectors."""
        if self.transform is not None:
            images = self.transform(images).unsqueeze(0)
        with torch.no_grad():

            if self.feat_type == 'cls':
                features = self.model(images.to(device))

            elif self.feat_type == 'patch':

                feat_out = {}

                def hook_fn_forward_qkv(module, input, output):
                    feat_out["qkv"] = output

                self.model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)

                # Forward pass in the model
                attentions = self.model.forward_selfattention(images.to(device))

                # Dimensions
                nb_im = attentions.shape[0]  # Batch size
                nh = attentions.shape[1]  # Number of heads
                nb_tokens = attentions.shape[2]  # Number of tokens

                # Extract the qkv features of the last attention layer
                qkv = (
                    feat_out["qkv"]
                        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                        .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0], qkv[1], qkv[2]
                k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
                q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
                v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]

                features = k

        return features
