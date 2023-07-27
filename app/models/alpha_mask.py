import torch
import torch.nn
import torch.nn.functional as F


class AlphaGridMask(torch.nn.Module):
    """
    A class for the alpha grid mask.

    Args:
        device (str): The device to use.
        aabb (torch.Tensor): The axis-aligned bounding box.
        alpha_volume (torch.Tensor): The alpha volume.
    """

    def __init__(self, device, aabb, alpha_volume):
        super().__init__()
        self.device = device

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1], alpha_volume.shape[-2], alpha_volume.shape[-3]]).to(
            self.device
        )

    def sample_alpha(self, xyz_sampled):
        """
        Samples the alpha values.

        Args:
            xyz_sampled (torch.Tensor): The sampled coordinates.

        Returns:
            torch.Tensor: The alpha values.
        """
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        """
        Normalizes the sampled coordinates.

        Args:
            xyz_sampled (torch.Tensor): The sampled coordinates.

        Returns:
            torch.Tensor: The normalized coordinates.
        """
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1
