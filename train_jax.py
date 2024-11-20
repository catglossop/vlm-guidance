# WARNING: importing tensorflow too late can silence important logging (╯°□°)╯︵ ┻━┻
import tensorflow as tf

# isort: split

import datetime
from functools import partial
import os
import os.path as osp

from absl import app, flags, logging
from flax.traverse_util import flatten_dict
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags
import optax
import tqdm
import wandb




def main(_):
    jax.utils.intialize_compliation_cache()

    