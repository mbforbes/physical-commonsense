"""General purpose utilities"""

import json

import numpy as np


def np2str(a: np.ndarray) -> str:
    """Turns a numpy array into a string. Use str2np to convert back.

    Yes, it is absurd that we need to do this. We shouldn't. But from my Expert
    Research TM (StackOverflow:
    https://stackoverflow.com/questions/35612235/how-to-read-numpy-2d-array-from-string)
    this was the best way.
    """
    return json.dumps(a.tolist())


def str2np(s: str) -> np.ndarray:
    """Turns a string into a numpy array. Reverse of np2str."""
    return np.array(json.loads(s))
