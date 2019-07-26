import numpy as np


def can_broadcast(s1, s2):
    s1a = np.asarray(s1)
    s2a = np.asarray(s2)
    return ((s1a == 1) | (s2a==1) | (s2a == s1a)).all()