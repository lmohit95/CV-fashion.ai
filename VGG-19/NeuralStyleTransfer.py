import tensorflow as tf
from ImageLoader import ImageLoader
from ImageResize import ImageResize
from StyleContentModel import StyleContentModel
import time

BLOCK1_CONV1 = "block1_conv1"
BLOCK2_CONV1 = "block2_conv1"
BLOCK3_CONV1 = "block3_conv1"
BLOCK4_CONV1 = "block4_conv1"
BLOCK5_CONV1 = "block5_conv1"
BLOCK5_CONV2 = "block5_conv2"


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


class NeuralStyleTransfer:
    def __init__(self, contentpath, stylepath, stylelayers, contentlayers):
        print("NST __init__")
        self.content_image = ImageLoader(ImageResize())(contentpath)
        self.style_image = ImageLoader(ImageResize())(stylepath)
        self.extractor = StyleContentModel(stylelayers, contentlayers)
        self.style_targets = self.extractor(self.style_image)['style']
        self.content_targets = self.extractor(self.content_image)['content']
        self.opt = tf.optimizers.Adam(learning_rate=0.005, beta_1=0.83, epsilon=1e-1)
        self.style_weight = 1e-2
        self.content_weight = 1e4

    def calculate_loss(self, outputs):
        print("NeuralStyleTransfer calculate_loss")
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        loss_style = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2)
                               for name in style_outputs.keys()])
        loss_style *= self.style_weight / len(style_layers)

        loss_content = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name]) ** 2)
                                 for name in content_outputs.keys()])
        loss_content *= self.content_weight / len(content_layers)
        return loss_style + loss_content

    @tf.function()
    def train_model(self, image):
        print("NeuralStyleTransfer train_model")
        total_variation_weight = 450
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.calculate_loss(outputs) + total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    def start_training(self, epochs=10, steps_per_epoch=100):
        print("NeuralStyleTransfer start_training")
        start = time.time()
        content_image = tf.Variable(self.content_image)

        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_model(content_image)
            print("Train step: {}".format(step))

        end = time.time()
        print("Total training time: {:.1f}".format(end - start))
        tf.keras.preprocessing.image.save_img('stylized-image.png', content_image[0])


# Set the paths for the content and style images
content_path = "content.jpg"
style_path = "style.jpg"

# 1st layer of all the blocks is used for style
# 2nd layer of 5th block is considered for content image
style_layers = [BLOCK1_CONV1, BLOCK2_CONV1, BLOCK3_CONV1, BLOCK4_CONV1, BLOCK5_CONV1]
content_layers = [BLOCK5_CONV2]

# Create an instance of NeuralStyleTransfer class
nst = NeuralStyleTransfer(content_path, style_path, style_layers, content_layers)
nst.start_training(epochs=20)
