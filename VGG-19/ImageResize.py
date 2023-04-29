import tensorflow as tf


class ImageResize:
    def __init__(self, max_dim=256):
        print("ImageResize __init__")
        self.max_dim = max_dim

    def __call__(self, image):
        print("ImageResize __call__: resizing image")
        shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        # Scale the image to a size proportional to its maximum dimension
        scaled_img = tf.cast(shape * (4 * self.max_dim / max(shape)), tf.int32)
        return tf.image.resize(image, scaled_img)
