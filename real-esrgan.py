import os
import time
import argparse
from typing import Optional

import cv2 as cv
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from colorama import Fore, Style
import tqdm
import streamlit as st

def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

class _NormBase(nn.Module):
    _version = 2
    __constants__ = ["track_running_stats", "momentum", "eps", "num_features", "affine"]
    num_features: int
    eps: float
    momentum: float
    affine: bool
    track_running_stats: bool
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_features, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_features, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, **factory_kwargs))
            self.register_buffer('running_var', torch.ones(num_features, **factory_kwargs))
            self.running_mean: Optional[torch.Tensor]
            self.running_var: Optional[torch.Tensor]
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
            self.num_batches_tracked: Optional[torch.Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # running_mean/running_var/num_batches... are registered at runtime depending
            # if self.track_running_stats is on
            self.running_mean.zero_()  # type: ignore[union-attr]
            self.running_var.fill_(1)  # type: ignore[union-attr]
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}".format(**self.__dict__)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )



class _BatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return nn.functional.batch_norm(
            input,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def interpolate(self, feat):
        return nn.functional.interpolate(feat, scale_factor=2, mode='nearest')

    def forward(self, x: torch.FloatTensor, mode: str='command'):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        
        feat = self.conv_first(feat)
        pipelines = [layer for layer in self.body]
        pipelines.extend([
            self.conv_body,
            lambda x : x + feat,
            self.interpolate,
            self.conv_up1,
            self.lrelu,
            self.interpolate,
            self.conv_up2,
            self.lrelu,
            self.conv_hr,
            self.lrelu,
            self.conv_last
        ])
        
        if mode != 'command':
            with st.sidebar:
                progress_bar = st.progress(value=0, text='rebuilding')
        
        count = 0        
        out = feat
        for layer_fn in tqdm.tqdm(pipelines, ncols=80, colour='green', desc='rebuilding'):
            out = layer_fn(out)
            if mode != 'command':
                count += 1
                percent = int(count / len(pipelines) * 100)
                progress_bar.progress(value=percent, text=f'rebuilding ({percent}%)')
        
        progress_bar.progress('Bingo 🐳. Just Download')
        return out


def resize(img : np.ndarray, height=None, width=None) -> np.ndarray:
    if height is None and width is None:
        raise ValueError("not None at the same time")
    if height is not None and width is not None:
        raise ValueError("not not None at the same time")
    h, w = img.shape[0], img.shape[1]
    if height:
        width = int(w / h * height)
    else:
        height = int(h / w * width)
    target_img = cv.resize(img, dsize=(width, height))
    return target_img

def show_image(img, winname = 'Default', height = None, width = None, format="bgr"):
    if format.lower() == "rgb":
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            
    cv.namedWindow(winname, cv.WINDOW_AUTOSIZE)
    if height or width:
        img = resize(img, height, width)
    cv.imshow(winname, img)
    cv.waitKey(0)
    cv.destroyWindow(winname)


def main(img_path : str, out_path: str, model_path: str, device: str, scale=4, outscale=None):
    model_state_dict = torch.load(model_path)
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        scale=scale
    )

    s = time.time()

    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()

    img = Image.open(img_path).convert('RGB')
    img = np.array(img) / 255
    h = img.shape[0]
    w = img.shape[1]
    img = torch.FloatTensor(img.transpose((2, 0, 1))).unsqueeze(0).to(device)
    
    s = time.time()
    with torch.no_grad():
        output = model(img)

    out_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    out_img = out_img[[2, 1, 0], :, :].transpose((1, 2, 0))
    
    out_img = out_img * 255
    out_img = out_img.round().astype('uint8')
    
    if outscale is not None:
        out_img = resize(out_img, height=int(h * outscale))

    img_full_name = os.path.split(img_path)[-1]
    img_name, ext_name = img_full_name.split(".")[0], img_full_name.split(".")[-1]
    
    if out_path != '-1':
        out_img_path = out_path
    else:
        out_img_path = "{}.out.{}".format(img_name, ext_name)
    print('Rebuild Image\'s shape', Fore.YELLOW, out_img.shape, Style.RESET_ALL)
    b = out_img[..., 0]
    g = out_img[..., 1]
    r = out_img[..., 2]
    out_img = np.stack([r, g, b], axis=2)
    Image.fromarray(out_img).save(out_img_path)
    cost_time = round(time.time() - s, 2)
    print('Result saved to', Fore.GREEN, out_img_path, Style.RESET_ALL, 'cost {} s'.format(cost_time))   


_model_instance = None

@st.cache_resource
def load_model_from_cache(model_path):
    global _model_instance
    if _model_instance is None:
        model_state_dict = torch.load(model_path)
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=4
        )
        model.load_state_dict(model_state_dict)
        _model_instance = model
        
    return model    

@st.cache_data
def streamlit_main(img_path, device: str, model_path):
    model = load_model_from_cache(model_path)
    model = model.to(device)
    model.eval()

    img = Image.open(img_path).convert('RGB')
    img = np.array(img) / 255
    h = img.shape[0]
    w = img.shape[1]
    img = torch.FloatTensor(img.transpose((2, 0, 1))).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img, mode='streamlit')
        
    out_img = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    out_img = out_img[[2, 1, 0], :, :].transpose((1, 2, 0))
    out_img = out_img * 255
    out_img = out_img.round().astype('uint8')
    
    b = out_img[..., 0]
    g = out_img[..., 1]
    r = out_img[..., 2]
    out_img = np.stack([r, g, b], axis=2)

    return Image.fromarray(out_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--device', default="cpu",
        help="device for inference"
    )

    parser.add_argument(
        '-i', 
        help="input image path"
    )

    parser.add_argument(
        '-o', default="-1",
        help="output image path"
    )

    parser.add_argument(
        '-m', '--model', 
        default='./model.bin',
        help="model path"
    )

    parser.add_argument(
        '-r', '--ratio', type=int, default=4,
        help="reinforcement resolution"
    )

    args = parser.parse_args()
    main(
        img_path=args.i,
        out_path=args.o,
        model_path=args.model,
        device=args.device,
        scale=args.ratio
    )
