import os

from dist_render.comm.singleton import SingletonMeta

from .utils import read_poses_city


class CityTestData(metaclass=SingletonMeta):
    """
    City dataset.
    """

    def __init__(self, args_nerf):
        self.root_dir = os.path.join(args_nerf.dataroot, args_nerf.datadir)
        self.downsample_train = args_nerf.downsample_train
        self.subfolder = args_nerf.subfolder
        self.poses = None
        self.img_hw = None
        self.get_poses()

    def get_poses(self):
        if self.poses is None:
            print("Read City poses")
            self.poses, self.img_hw = read_poses_city(self.root_dir, self.downsample_train, self.subfolder)
        return self.poses
