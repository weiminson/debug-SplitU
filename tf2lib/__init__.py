import tensorflow as tf

from tf2lib.data import *
from tf2lib.image import *
from tf2lib.ops import *
from tf2lib.utils import *
version = 0

if version == 0:

    tf.config.gpu.set_per_process_memory_growth(enabled=True)

if version == 1:
    # define gpu in a new version
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # Memory growth must be set at program startup
            print(e)