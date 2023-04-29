import tensorflow as tf
from GramMatrix import GramMatrix
from VGGModel import VGGModel


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        print("StyleContentModel __init__")
        super(StyleContentModel, self).__init__()
        # Load a VGGModel with specified style and content layers
        self.vgg = VGGModel(style_layers + content_layers).model
        self.vgg.trainable = False
        self.style_layers = style_layers
        self.content_layers = content_layers

    def __call__(self, inputs):
        print("StyleContentModel __call__")
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:len(self.style_layers)], outputs[len(self.style_layers):])
        style_outputs = [GramMatrix()(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}
