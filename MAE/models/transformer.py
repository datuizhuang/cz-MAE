from timm.models.vision_transformer import VisionTransformer, Block
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class Decoder(nn.Module):
    '''
    MAE中的解码器
    input：所有的patch（mask部分的patch为可学习的params）
    output：通过解码器重建后的被mask的部分的值
    '''

    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 num_patches=196):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * patch_size ** 2

        self.num_features = embed_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = num_patches

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_token_num=0):
        for block in self.blocks:
            x = block(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:, :]))
        else:
            x = self.head(self.norm(x))
        return x  # [batch, num_mask, num_classes(3 * patch_size ** 2)]


class Encoder(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4, qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, act_layer=None):
        super(Encoder, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      num_classes=num_classes,
                                      embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                      drop_path_rate=drop_path_rate, norm_layer=norm_layer, act_layer=act_layer)

        self.num_patches = self.patch_embed.num_patches
        # 很多patch被mask掉了，因此必须要有pos embedding；,论文利用sin-cos形式
        self.pos_embed.data = get_sinusoid_encoding_table(self.num_patches, embed_dim)

    def forward_features(self, x, mask: torch.Tensor):
        x = self.patch_embed(x)
        # 采用了sin-cos形式，而本身pos embed是nn.parameter，因此这里detach()
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        b, p, d = x.size()
        x_visible = x[~mask].reshape(b, -1, d)
        x = self.blocks(x_visible)
        x = self.norm(x)
        return x

    def forward(self, x, mask: torch.Tensor):
        x = self.forward_features(x, mask)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=0, decoder_num_classes=768,
                 embed_dim=768, decoder_embed_dim=512,
                 depth=12, decoder_depth=8,
                 num_heads=12, decoder_num_heads=8,
                 mlp_ratio=4, qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, act_layer=None):
        super(EncoderDecoder, self).__init__()

        # 创建编码器和解码器
        self.encoder = Encoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                               num_classes=num_classes, embed_dim=embed_dim,
                               depth=depth, num_heads=num_heads, qkv_bias=qkv_bias, mlp_ratio=mlp_ratio,
                               drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                               drop_path_rate=drop_path_rate,
                               norm_layer=norm_layer, act_layer=act_layer)

        self.decoder = Decoder(patch_size=patch_size, num_classes=decoder_num_classes, embed_dim=decoder_embed_dim,
                               depth=decoder_depth, num_heads=decoder_num_heads,
                               mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
                               attn_drop_rate=attn_drop_rate,
                               drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                               num_patches=self.encoder.patch_embed.num_patches)

        self.encoder_to_decoder = nn.Linear(embed_dim, decoder_embed_dim)
        self.pos_embedding = get_sinusoid_encoding_table(self.encoder.num_patches, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

    def forward_encoder(self, x):
        return self.encoder(x)

    def forward_decoder(self, x, return_token_num=0):
        x = self.encoder_to_decoder(x)
        return self.decoder(x, return_token_num)

    def forward(self, x, mask):
        x_visible = self.encoder(x, mask)  # [batch, vis, encoder_embed_dim]
        x_visible = self.encoder_to_decoder(x_visible)  # [batch, vis, decoder_embed_dim]

        b, v, d = x_visible.size()

        pos_embed = self.pos_embedding.expand(b, -1, -1).to(x.device).clone().detach()
        pos_embed_vis = pos_embed[~mask].reshape(b, -1, d)
        pos_embed_mask = pos_embed[mask].reshape(b, -1, d)

        x_decoder = torch.cat([x_visible + pos_embed_vis, self.mask_token + pos_embed_mask], dim=1)
        num_masked = pos_embed_mask.size(1)

        decoder_output = self.decoder(x_decoder, num_masked)
        return decoder_output  # [batch, num_mask, 3 * patch_size**2]


def test_decoder():
    model = Decoder(num_patches=9, embed_dim=768)
    x = torch.rand(1, 9, 768)
    y = model(x, 0)
    print(y.size())


def gene_mask(num_patch, ratio):
    num_mask = int(num_patch * ratio)

    zero = np.zeros(num_patch - num_mask)
    one = np.ones(num_mask)
    mask = np.concatenate((zero, one), axis=0)
    np.random.shuffle(mask)
    mask = torch.from_numpy(mask)
    mask = mask.view(1, -1)
    return mask.to(torch.bool)


def test_encoder_decoder():
    img_size = 224
    patch_size = 16
    num_patch = int(img_size // patch_size) ** 2

    model = EncoderDecoder(depth=2, decoder_depth=2, patch_size=patch_size)
    x = torch.rand(1, 3, img_size, img_size)
    mask = gene_mask(num_patch, 0.75)

    y = model(x, mask)
    print(y.size())


if __name__ == '__main__':
    test_encoder_decoder()
