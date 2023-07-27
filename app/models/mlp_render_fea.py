import torch
import torch.nn
from tools.utils import positional_encoding


class MLPRender_Fea(torch.nn.Module):
    """
    A MLP for rendering feature

    Args:
        inChanel (int): The number of input channels.
        viewpe (int): The number of positional encoding dimensions for the view direction.
        feape (int): The number of positional encoding dimensions for the input features.
        featureC (int): The number of output channels.
        bias_enable (bool): Whether to enable bias.
    """

    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128, bias_enable=False):
        super().__init__()
        self.in_mlpC = 2 * max(viewpe, 0) * 3 + 2 * feape * inChanel + 3 * (viewpe > -1) + inChanel
        self.viewpe = viewpe
        self.feape = feape

        layer1 = torch.nn.Linear(self.in_mlpC, featureC, bias=bias_enable)
        layer2 = torch.nn.Linear(featureC, featureC, bias=bias_enable)
        layer3 = torch.nn.Linear(featureC, 3, bias=bias_enable)

        self.mlp = torch.nn.Sequential(
            layer1,
            torch.nn.ReLU(inplace=True),
            layer2,
            torch.nn.ReLU(inplace=True),
            layer3,
        )
        if bias_enable:
            torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, viewdirs, features):
        """
        Forward pass of the MLP Render Feature.

        Args:
            viewdirs (torch.Tensor): The view direction tensor.
            features (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The output RGB tensor.
        """
        indata = [features]
        if self.viewpe > -1:
            indata += [viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)
        return rgb
