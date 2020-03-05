import torchvision.transforms.functional as tf


class PadToSquare:
    def __init__(self):
        pass

    def __call__(self, x):
        width, height = x.size
        pad = int(abs(width - height) / 2)
        if pad != 0:
            if width > height:
                return tf.pad(x, padding=(0, pad, 0, pad), fill=(255, 255, 255))
            else:
                return tf.pad(x, padding=(pad, 0, pad, 0), fill=(255, 255, 255))
        else:
            return x
