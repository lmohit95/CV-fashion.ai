import tensorflow as tf


class VGGModel:
    def __init__(self, layer_names):
        print("VGGModel __init__")
        self.vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        self.vgg.trainable = False
        # Get the output tensors from the specified layers
        self.outputs = [self.vgg.get_layer(name).output for name in layer_names]
        self.model = tf.keras.Model([self.vgg.input], self.outputs)
