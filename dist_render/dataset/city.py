import os

from comm.singleton import SingletonMeta

from .utils import read_poses_city


class CityTestData(metaclass=SingletonMeta):
    """
    city dataset.
    """

    def __init__(self, args_nerf):
        self.datadir = args_nerf.datadir
        self.root_dir = os.path.join(args_nerf.dataroot, self.datadir)
        self.downsample_train = args_nerf.downsample_train
        self.partition = args_nerf.partition
        self.filter_ray = args_nerf.filter_ray
        self.poses = None
        self.img_hw = None
        self.get_poses()

    def get_poses(self, all_block_render_test=False):
        """
        get dataset poses from datadir in config.

        Args:
            all_block_render_test(bool): read all block poses.

        Returns:
            list: poses.
        """
        if self.poses is None:
            print("Read Shcity poses, all_block_render_test=", all_block_render_test)
            self.poses, self.img_hw = read_poses_city(
                self.datadir,
                self.root_dir,
                self.downsample_train,
                self.partition,
                self.filter_ray,
                all_block_render_test,
            )
        return self.poses


class CityPartTestData(CityTestData):
    """
    City part dataset.
    """

    def get_poses(self):
        return super().get_poses(all_block_render_test=False)


class CityAllBlockTestData(CityTestData):
    """
    City all block dataset.
    """

    def get_poses(self):
        return super().get_poses(all_block_render_test=True)
