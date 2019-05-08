import random
import numpy as np
import tensorflow as tf
from collections import OrderedDict, defaultdict

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

class OrderedDefaultDict(OrderedDict):
    def __missing__(self, key):
        self[key] = defaultdict(list)
        return self[key]

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def fmt_row(width, row, header=False):
    out = " | ".join(fmt_item(x, width) for x in row)
    if header: out = out + "\n" + "-"*len(out)
    return out

def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim == 0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep
