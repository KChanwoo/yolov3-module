"""
GPU setting
:author ChanwooKwon, IOSYS (c) AI Lab, (2020.01.~2020.03.13)
"""

import tensorflow as tf

def initialize_gpu(gpu_index=-1, memory_size=-1):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if memory_size == -1:
        if gpus:
            if gpu_index == -1:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            elif gpu_index >= len(gpus):
                raise Exception("GPU Out Of Index Exception")
            else:
                tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
    else:
        if gpus:
            if gpu_index == -1:
                raise Exception("Can not allocate memory to all gpus.")
            elif gpu_index >= len(gpus):
                raise Exception("GPU Out Of Index Exception")
            else:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[gpu_index],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_size)])