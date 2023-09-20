# -*- coding: utf-8 -*-


class Task(object):
    """
    base task defination.
    """

    def __init__(self, uid=None, value=None):
        self.uid = uid
        self.value = value


class EmptyTask(Task):
    """
    empty task defination
    """

    def __init__(self):
        super().__init__()


class StopTask(Task):
    """
    stop task defination
    """

    def __init__(self):
        super().__init__()
