from typing import Any, List, Tuple


class PrintableConfig:
    """Printable Config defining str function"""

    def __init__(self, args=None) -> None:
        if args is not None:
            self.parse_from_args(args)

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)

    def parse_from_args(self, args):
        """This means to parse from config_parser args"""
        for k, v in args.__dict__.items():
            self.set(k, v)

    def parse_from_kwargs(self, **kwargs):
        """
        Parse configs from kwargs.
        """
        for k, v in kwargs.items():
            self.set(k, v)

    def set(self, name, value):
        assert not hasattr(self, name)
        setattr(self, name, value)

    def save_config(self, file_path):
        with open(file_path, "w", encoding="utf-8") as file:
            for attr_name in sorted(vars(self)):
                attr_val = getattr(self, attr_name)
                file.write(f"{attr_name} = {attr_val}\n")


class ArgsConfig(PrintableConfig):
    """
    Configs parsed from args.

    Example:
    class ArgsDemo1:
        def __init__(self) -> None:
            self.a = 1
            self.b = "2"

    class ArgsDemo2:
        def __init__(self) -> None:
            self.c = 11
            self.d = "22"

    arg1 = ArgsDemo1()
    arg2 = ArgsDemo2()
    training_config = ArgsConfig([arg1, arg2])
    print(str(training_config))
    training_config.a = 4
    training_config.e = 6
    print(str(training_config))

    config_saving_path = "~/landmark/test_config.txt"
    training_config.save_config(config_saving_path)
    """

    IGNORE_ATTR_NAME = "config_list"

    def __init__(self, args_list: List) -> None:
        super().__init__()
        self.config_list = [PrintableConfig(args) for args in args_list]

    def save_config(self, file_path):
        all_attrs = vars(self)
        for config in self.config_list:
            all_attrs.update(vars(config))

        with open(file_path, "w", encoding="utf-8") as file:
            for attr_name in sorted(all_attrs):
                if attr_name is self.IGNORE_ATTR_NAME:
                    continue
                file.write(f"{attr_name} = {all_attrs[attr_name]}\n")

    def __getattr__(self, __name: str) -> Any:
        if __name in self.__dict__:
            return self.__dict__[__name]

        for config in self.config_list:
            if __name in config.__dict__:
                return config.__dict__[__name]

    def __setattr__(self, __name: str, __value: Any) -> None:
        if self.IGNORE_ATTR_NAME in self.__dict__:
            for config in self.config_list:
                if __name in config.__dict__:
                    config.__dict__[__name] = __value
                    return

        self.__dict__[__name] = __value

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if key is self.IGNORE_ATTR_NAME:
                continue
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")

        for config in self.config_list:
            lines.append(str(config))
        return "\n    ".join(lines)
