import threading

from dist_render.comm.singleton import SingletonMeta


class GlobalArgsManager(metaclass=SingletonMeta):
    """
    Global args manager to change model args when engine is running.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.global_args = {"app_code": 2930, "edit_mode": 0}  # [edit_mode] 0: resetBuild, 1: newBuild, 2: removeBuild

    def get_arg(self, name):
        """
        get nerf changeable args.

        Args:
            name(str): args name.

        Returns:
            int: args value.
        """
        with self.lock:
            return self.global_args[name]

    def set_arg(self, name, value):
        """
        set nerf changeable args value.

        Args:
            name(str): args name.
            value(int): args value.
        """
        with self.lock:
            self.global_args[name] = value
