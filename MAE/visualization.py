from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def post_process(img):
    img.squeeze_()
    mask_image = np.array(img.clone().detach())

    mean = np.array([0.485, 0.456, 0.406])[None, None, :]
    std = np.array([0.229, 0.224, 0.225])[None, None, :]
    mask_image = (mask_image * std + mean)
    mask_image = np.clip(mask_image, a_max=1, a_min=0)
    return mask_image


def get_mask_image(img, mask):
    patch_size = 16
    b, c, h, w = img.size()

    image = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)  # (b, n, p, d)
    mask_image = image * (~mask[:, :, None, None])
    mask_image = rearrange(mask_image, 'b (h w) (p1 p2) c -> b (h p1) (w p2) c', h=h // patch_size, w=w // patch_size, p1=patch_size, p2=patch_size)
    mask_image = post_process(mask_image)
    plt.imsave('mask.jpg', mask_image)


def get_new_image(img, output, mask):
    '''
    Args:
        img:    [b, c, h, w]
        output: [b, num_mask, patch_size*patch_size*3]
        mask:   [b, num_patch], 0:vis, 1: mask
    '''
    patch_size = 16
    b, c, h, w = img.size()

    patch_image = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)  # [b, num_patch, patch_size*patch_size*3]
    patch_image = post_process(patch_image)
    patch_image = torch.from_numpy(patch_image).unsqueeze(0).to(torch.float32)
    patch_image = rearrange(patch_image, 'b n p c -> b n (p c)')
    patch_image[mask] = output.to(torch.float32)
    patch_image = rearrange(patch_image, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', c=c, h=h // patch_size, p1=patch_size)
    patch_image.squeeze_()
    patch_image = np.array(patch_image.clone().detach()).clip(min=0, max=1)

    plt.imsave('reconstruction.jpg', patch_image)


def build_model():
    img_size = 224
    patch_size = 16
    in_chans = 3
    encoder_num_classes = 0
    decoder_num_classes = 3 * 16 ** 2
    encoder_embed_dim = 1024
    decoder_embed_dim = 512
    encoder_depth = 8
    decoder_depth = 6
    encoder_heads = 8
    decoder_heads = 4
    mlp_ratio = 4
    drop_rate = 0
    attn_drop_rate = 0
    drop_path_rate = 0
    encoder_qkv_bias = True
    decoder_qkv_bias = False
    model = EncoderToDecoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                    encoder_num_classes=encoder_num_classes, decoder_num_classes=decoder_num_classes,
                                    encoder_embed_dim=encoder_embed_dim, decoder_embed_dim=decoder_embed_dim,
                                    encoder_depth=encoder_depth, decoder_depth=decoder_depth,
                                    encoder_heads=encoder_heads, decoder_heads=decoder_heads,
                                    mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                    encoder_qkv_bias=encoder_qkv_bias, decoder_qkv_bias=decoder_qkv_bias)
    model_dic = torch.load('epoch_600.pth', map_location='cpu')['model']
    new_dic = {}
    for key, value in model_dic.items():
        if 'module' in key:
            key = key[7:]
        new_dic[key] = value
    model.load_state_dict(new_dic)

    return model


def getTrans():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet/v1
    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    return train_transform


def generate_mask():
    mask_num = int(196 * 0.75)
    num_patch = 196

    mask = np.concatenate([np.zeros(num_patch - mask_num, dtype=np.bool), np.ones(mask_num, dtype=np.bool)], axis=0)
    np.random.shuffle(mask)
    return mask


def get_image(img_path: str = ""):
    image = Image.open(img_path).convert('RGB')
    transform = getTrans()
    image = transform(image)
    image.unsqueeze_(0)
    mask = generate_mask()
    mask = torch.from_numpy(mask).unsqueeze(0)
    return image, mask


def test():
    img_path = r'./cat.jpg'
    img, mask = get_image(img_path)
    print(img.size())
    get_mask_image(img, mask)

    model = build_model().eval()
    output = model(img, mask)

    get_new_image(img, output, mask)


if __name__ == '__main__':
    test()
