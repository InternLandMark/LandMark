import torch
import torch.nn.functional as F
from tools.utils import positional_encoding
from torch import nn


def raw2alpha(raw, dists, act_fn=F.relu):
    """
    Computes the alpha values from the raw values and distances.

    Args:
        raw (torch.Tensor): The raw values.
        dists (torch.Tensor): The distances.
        act_fn (function): The activation function. Default is F.relu.

    Returns:
        torch.Tensor: The alpha values.
    """
    return 1.0 - torch.exp(-act_fn(raw) * dists)


def raw2outputs(raw, dists=1):
    """
    Computes the outputs from the raw values and distances.

    Args:
        raw (torch.Tensor): The raw values.
        dists (torch.Tensor): The distances.

    Returns:
        dict: A dictionary containing the RGB map, weights, RGB values, and alpha values.
    """
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    alpha = raw2alpha(raw[..., 3], dists)  # .unsqueeze(-1)
    weights = (
        alpha
        * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 1.0 - alpha + 1e-10], -1),
            -1,
        )[:, :-1]
    )
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    return {
        "rgb_map": rgb_map,
        "weights": weights,
        "rgb": rgb,
        "alpha": alpha,
    }


class NeRF(torch.nn.Module):
    """
    Class for NeRF branch
    """

    def __init__(self, args, den_n_comp, app_n_comp, nfreqs=None):  # pure frequency embed
        super().__init__()

        self.num_freq = args.nerf_freq if nfreqs is None else nfreqs
        self.den_n_comp = den_n_comp
        self.app_n_comp = app_n_comp
        self.skips = [4]
        self.use_viewdirs = True

        self.D = args.nerf_D
        self.D_a = args.nerf_D_a
        self.W = args.nerf_W

        self.init_module_v0()

    def init_module_v0(self):
        W, D, D_a = self.W, self.D, self.D_a
        input_ch = 3  # pts
        input_ch += 2 * 3 * self.num_freq  # pts_pe
        input_ch += self.den_n_comp  # 3 planes, 8 component per plane

        input_ch_views = 3  # views
        input_ch_views += 2 * 3 * 4  # views_pe
        input_ch_views += self.app_n_comp

        self.pts_linears = torch.nn.ModuleList(
            [torch.nn.Linear(input_ch, W)]
            + [torch.nn.Linear(W, W) if i not in self.skips else torch.nn.Linear(W + input_ch, W) for i in range(D - 1)]
        )

        self.views_linears = nn.ModuleList(
            [torch.nn.Linear(input_ch_views + W, W // 2)] + [torch.nn.Linear(W // 2, W // 2) for i in range(D_a - 1)]
        )

        if self.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, 4)

    def forward(self, pts, viewdir=None, den_feats=None, app_feats=None, dists=None):  # gridfeat store in dict format
        """
        Forward pass of the NeRF branch.

        Args:
            pts (torch.Tensor): The input points tensor.
            viewdir (torch.Tensor): The view direction tensor. Default is None.
            den_feats (torch.Tensor): The density field tensor. Default is None.
            app_feats (torch.Tensor): The appearance field tensor. Default is None.
            dists (torch.Tensor): The distances. Default is None.

        Returns:
            dict: A dictionary containing the RGB map, weights, RGB values, and alpha values.
        """
        nray, npts = pts.shape[:2]
        pts = pts.view(-1, 3)
        pts_pe = positional_encoding(pts, self.num_freq)
        if viewdir is not None:
            viewdir = viewdir.reshape(-1, 3)  # use reshape when sample multiple pts
            viewdir_pe = positional_encoding(viewdir, 4)
        input_concat = [pts, pts_pe]
        input_concat += [den_feats]
        inputs_flat = torch.cat(input_concat, -1)  # concat
        h = inputs_flat
        for i, _ in enumerate(self.pts_linears):
            h = F.relu(self.pts_linears[i](h))
            if i in self.skips:
                h = torch.cat([inputs_flat, h], -1)
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            view_concat = [feature, viewdir, viewdir_pe]
            view_concat += [app_feats]
            h = torch.cat(view_concat, -1)
            for i, _ in enumerate(self.views_linears):
                h = F.relu(self.views_linears[i](h))
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return raw2outputs(outputs.view(nray, npts, -1), dists)


class NeRFParallel(NeRF):
    """
    A DDP version of NeRF branch
    """

    def __init__(self, args, den_n_comp, app_n_comp, nfreqs=None):  # pure frequency embed
        super().__init__(args, den_n_comp, app_n_comp, nfreqs)

    def forward(self, pts, viewdir=None, den_feats=None, app_feats=None, dists=None):
        """
        Forward pass of the NeRF branch.

        Args:
            pts (torch.Tensor): The input points tensor.
            viewdir (torch.Tensor): The view direction tensor. Default is None.
            den_feats (torch.Tensor): The density field tensor. Default is None.
            app_feats (torch.Tensor): The appearance field tensor. Default is None.
            dists (torch.Tensor): The distances. Default is None.

        Returns:
            dict: A dictionary containing the RGB map, weights, RGB values, and alpha values.
        """
        pts_pe = positional_encoding(pts, self.num_freq)
        if viewdir is not None:
            viewdir_pe = positional_encoding(viewdir, 4)
        input_concat = [pts, pts_pe]
        input_concat += [den_feats]
        inputs_flat = torch.cat(input_concat, -1)  # concat
        h = inputs_flat
        for i, _ in enumerate(self.pts_linears):
            h = F.relu(self.pts_linears[i](h))
            if i in self.skips:
                h = torch.cat([inputs_flat, h], -1)
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            view_concat = [feature, viewdir, viewdir_pe]
            view_concat += [app_feats]
            h = torch.cat(view_concat, -1)
            for i, _ in enumerate(self.views_linears):
                h = F.relu(self.views_linears[i](h))
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
