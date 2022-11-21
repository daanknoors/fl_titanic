"""Utility functions"""
import numpy as np
import pandas as pd

from src import config


def simplified_repr(self):
    """Simplified repr message for displaying all types of parameters in instance.
    Instance might have classes as parameters. Parameters within these classes are
    not displayed to reduce output length"""
    msg = f"{self.__class__.__name__}("

    dict_vars = vars(self)
    if not dict_vars:
        msg += ')'
        return msg

    msg += "\n"
    for i, (k, v) in enumerate(dict_vars.items()):
        # if parameter is a class, don't show all parameters
        if hasattr(v, '__dict__'):
            msg += f"       {k}={v.__class__.__name__}()"
        # if parameter is a dictionary
        elif isinstance(v, dict):
            msg += f"       {k}="
            msg += "{"

            dict_vars2 = v
            for i2, (k2, v2) in enumerate(dict_vars2.items()):
                if hasattr(v2, '__dict__'):
                    msg += f"{k2}={v2.__class__.__name__}()"
                else:
                    msg += f"{k}={v}"

                if i2 != len(dict_vars2) - 1:
                    msg += ', '
            msg += "}"
        # if parameter is a list
        elif isinstance(v, list):
            msg += f"       {k}=["
            for i3, l in enumerate(v):
                if hasattr(l, '__dict__'):
                    msg += f"{l.__class__.__name__}()"
                else:
                    msg += f"{l}"
                if i3 != len(v) - 1:
                    msg += ', '
            msg += f"]"
        # if parameter is value
        else:
            msg += f"       {k}={v}"

        if i != len(dict_vars)-1:
            msg += ', \n'

    msg += "\n)\n"
    # attributes = ", ".join([f"{k}={v}" for k, v in vars(self).items()])
    return msg