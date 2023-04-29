import tensorflow as tf


class ImageLoader:
    def __init__(self, scaler):
        self.scaler = scaler

    def __call__(self, path_to_img):
        print("ImageLoader __call__")
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return self.scaler(img)[tf.newaxis, :]

