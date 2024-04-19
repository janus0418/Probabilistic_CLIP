""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size
        )
        act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            global_average_pool=vision_cfg.global_average_pool,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            pretrained=text_cfg.hf_model_pretrained
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text

############ Added by me 09/12 ############
DIM = 1024
MLP_DEPTH = 6
class MLP(nn.Module):
    def __init__(self, is_Sigma = False, input_size=DIM, hidden_size=DIM, output_size=DIM, depth = MLP_DEPTH):
        super(MLP, self).__init__()
        self.mid_layer_num = max(0,MLP_DEPTH-2)
        self.is_Sigma = is_Sigma
        layers = []
        self.m = nn.Softplus(beta=1, threshold= 20)
        # self.m = nn.ReLU()
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(self.mid_layer_num):
            layers.append(nn.Linear(hidden_size,hidden_size))
            layers.append(nn.ReLU())


        layers.append(nn.Linear(hidden_size,output_size))

        self.middle_layers = nn.Sequential(*layers)
        # ## JIAHAO EDITS 
        # # self.middle_layers = nn.ModuleList(layers)
        # for name, module in self.named_modules():
        #     # check if name ends with .10
        #     # check if the rank of the current gpu is 1
        #         # print("rank 0 has been hit 189")
        #         if ".10" in name or "m" == name: # name.endswith(f".{self.mid_layer_num*2+1}"):
        #             if torch.distributed.get_rank() == 0:
        #                 print("#####line 190 SOFTPLUS hit" + name)
        #             def hook_fn(module, input, output,name=name):
        #                 if not isinstance(output, tuple):  # Ensure output is a tuple
        #                     output = (output,)
        #                 if not isinstance(input, tuple):  # Ensure output is a tuple
        #                     input = (input,)

        #                 for idx, tuple_val in enumerate(input):
        #                     if torch.distributed.get_rank() == 0:
        #                         if tuple_val is not None:
        #                         # if tuple_val is not None and torch.isnan(tuple_val).any():
        #                             min = tuple_val.min()
        #                             max = tuple_val.max() 
        #                             # print("**** 201 INPUT" +f"{module} of {name} {idx}!"+f"Min: {min}, Max: {max}")

        #                 for idx, tuple_val in enumerate(output):
        #                     if torch.distributed.get_rank() == 0:
        #                         if tuple_val is not None:
        #                             min = tuple_val.min()
        #                             max = tuple_val.max() 
        #                             # print("**** 207 OUTPUT "+f"{module} of {name}!"+f"Min: {min}, Max: {max}")                            
        #                     # if tuple_val is not None and torch.isnan(tuple_val).any():
        #                         # print("#### model 198 OUTPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at output index {idx}!")
        #                         # for idx, tuple_val_input in enumerate(input):
        #                         #     if tuple_val_input is not None:
        #                         #     # if tuple_val_input is not None and torch.isnan(tuple_val).any():
        #                         #         min = tuple_val_input.min()
        #                         #         max = tuple_val_input.max() 
        #                         #         print("**** 212 INPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at input index {idx}!"+f"Min: {min}, Max: {max}")
        #                         # raise ValueError(f"####### model 204; NaN value detected in gradients Stopping training.")

        #                 # for idx, tuple_val in enumerate(input):
        #                 #     tuple_val.fill_(0.01)
        #             module.register_backward_hook(hook_fn)    

        #         # for idx, tuple_val in enumerate(input):
        #         #     if tuple_val is not None and torch.isnan(tuple_val).any():
        #         #         min = tuple_val.min()
        #         #         max = tuple_val.max() 
        #         #         print("**** INPUT NaN value detected in gradients. Stopping training."+f"{module} of {name} contains NaN values at input index {idx}!"+f"Min: {min}, Max: {max}")



    def forward(self, x):
        # for layer_idx, layer in enumerate(self.middle_layers):
        #     x = layer(x)
        #     min = x.min().item()
        #     max = x.max().item()
        #     print("layer_idx: ", layer_idx, "min: ", min, "max: ", max)
        #     if torch.isnan(x).any():
        #         raise ValueError("##### NaN x at layer_idx: ", layer_idx)
        # quit()
        ### Edit with Jiahao
        x = self.middle_layers(x)
        if self.is_Sigma:       
            # x = self.m(x) + 0.000001
            x = self.m(x+0.5) + 0.000001
            # tryna fit the number to be in the range of 0.9 to 1.1 during init
            # --> prevent number from becoming very small during training

        return x
    
############ Added by me 09/12 ############



class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.MU_Images = MLP().to(device)
        # self.SIGMA_Images = MLP(is_Sigma = True).to(device)

        # self.MU_Texts = MLP().to(device)
        # self.SIGMA_Texts = MLP(is_Sigma = True).to(device)
        self.MU_Images = MLP()
        self.SIGMA_Images = MLP(is_Sigma = True)

        self.MU_Texts = MLP()
        self.SIGMA_Texts = MLP(is_Sigma = True)

        #################################################################
        # Commented out by me
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
        #################################################################


    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x


    ############ Added by me 09/26 ############
    #### Purpose: To use for zero_shot.py #####
    def mu_sigma_img(self, text_features,device):
        d = text_features.shape[1]
        batch_size = text_features.shape[0]
        mu_img = torch.empty(0, d).to(device)
        sigma_img = torch.empty(0, d).to(device)

        for i in range(batch_size):
            new_mu_img = self.MU_Images(text_features[i]).reshape(1,d)
            if torch.any(torch.isnan(new_mu_img)):
                raise ValueError("NaN new_mu_img")
            mu_img = torch.cat((mu_img, new_mu_img.to(device)), dim=0)

            new_sigma_img = self.SIGMA_Images(text_features[i]).reshape(1,d)
            if torch.any(torch.isnan(new_sigma_img)):
                raise ValueError("NaN new_sigma_img")
            sigma_img = torch.cat((sigma_img, new_sigma_img.to(device)), dim=0)
        return mu_img, sigma_img

    def mu_sigma_txt(self, image_features,device):
        d = image_features.shape[1]
        batch_size = image_features.shape[0]
        mu_txt = torch.empty(0, d).to(device)
        sigma_txt = torch.empty(0, d).to(device)

        for i in range(batch_size):
            new_mu_txt = self.MU_Texts(image_features[i]).reshape(1,d)
            if torch.any(torch.isnan(new_mu_txt)):
                raise ValueError("NaN new_mu_txt")
            mu_txt = torch.cat((mu_txt, new_mu_txt.to(device)), dim=0)

            new_sigma_txt = self.SIGMA_Texts(image_features[i]).reshape(1,d)
            if torch.any(torch.isnan(new_sigma_txt)):
                raise ValueError("NaN new_sigma_txt")
            sigma_txt = torch.cat((sigma_txt, new_sigma_txt.to(device)), dim=0)

        return mu_txt, sigma_txt  

    ############ Added by me 09/12 ############
    def mu_sigma_img_txt(self, image_features, text_features,device):
        d = image_features.shape[1]
        mu_img = torch.empty(0, d).to(device)
        sigma_img = torch.empty(0, d).to(device)
        mu_txt = torch.empty(0, d).to(device)
        sigma_txt = torch.empty(0, d).to(device)

        for i in range(image_features.shape[0]):
            new_mu_img = self.MU_Images(text_features[i]).reshape(1,d)
            if torch.any(torch.isnan(new_mu_img)):
                raise ValueError("NaN new_mu_img")
            mu_img = torch.cat((mu_img, new_mu_img.to(device)), dim=0)

            new_sigma_img = self.SIGMA_Images(text_features[i]).reshape(1,d)
            if torch.any(torch.isnan(new_sigma_img)):
                raise ValueError("NaN new_sigma_img")
            sigma_img = torch.cat((sigma_img, new_sigma_img.to(device)), dim=0)

            new_mu_txt = self.MU_Texts(image_features[i]).reshape(1,d)
            if torch.any(torch.isnan(new_mu_txt)):
                raise ValueError("NaN new_mu_txt")
            mu_txt = torch.cat((mu_txt, new_mu_txt.to(device)), dim=0)

            new_sigma_txt = self.SIGMA_Texts(image_features[i]).reshape(1,d)
            if torch.any(torch.isnan(new_sigma_txt)):
                raise ValueError("NaN new_sigma_txt")
            sigma_txt = torch.cat((sigma_txt, new_sigma_txt.to(device)), dim=0)

        return mu_img, sigma_img, mu_txt, sigma_txt
    ############ Added by me 09/12 ############


    def forward(self, image, text,device):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)

        has_nan_img_features = torch.any(torch.isnan(image_features))
        has_nan_txt_features = torch.any(torch.isnan(text_features))
        if has_nan_img_features and has_nan_txt_features:
            raise ValueError("NaN img,txt features")
        if has_nan_txt_features:
            raise ValueError("NaN txt features")
        if has_nan_img_features:
            raise ValueError("NaN img features")

        mu_img, sigma_img, mu_txt, sigma_txt = self.mu_sigma_img_txt(image_features, text_features,device)
        # return image_features, text_features, self.logit_scale.exp()
        return image_features, text_features, self.logit_scale.exp(), mu_img, sigma_img, mu_txt, sigma_txt


class CustomTextCLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed
