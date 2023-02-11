import inspect
from typing import Dict, Any

import pandas as pd


def update_functional(args: Dict[str, Any]):
    if args is None:
        return lambda x, y: x

    def f(default: Any, name: str):
        if name in args:
            return args[name]
        return default

    return f


class no_train(object):
    def __init__(self, module):
        self.module = module

    def __enter__(self):
        self.module.train(False)

    def __exit__(self):
        self.module.train(True)


def vprint(verbose=True):
    def f(x):
        if verbose:
            print(x)

    return f


def key_replace(dc, pattern):
    return {k.replace(pattern, ''): v for k, v in dc.items()}


def dict_print(d):
    print(pd.DataFrame([d]))


def has_arg(func, arg):
    return arg in inspect.signature(func).parameters
