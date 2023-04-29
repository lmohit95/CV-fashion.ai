import cv2
from numpy import load
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
from numpy import savez_compressed
from keras.models import load_model
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from tensorflow.keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
import keras.backend as K
import time

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


# Discrminiator
def create_discriminator(image_shape):
    # Initialize weights of the network with a normal distribution
	weight_init = RandomNormal(stddev=0.02)
	
	# Define inputs for the discriminator network
	source_image = Input(shape=image_shape)
	target_image = Input(shape=image_shape)
	
	# Concatenate source and target images as input to the network
	merged_images = Concatenate()([source_image, target_image])
	
	# Define layers of the discriminator network
	x = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(merged_images)
	x = LeakyReLU(alpha=0.2)(x)
	
	x = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	
	x = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	
	x = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	
	x = Conv2D(512, (4,4), padding='same', kernel_initializer=weight_init)(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.2)(x)
	
	x = Conv2D(1, (4,4), padding='same', kernel_initializer=weight_init)(x)
	patch_output = Activation('sigmoid')(x)
	
	# Define the discriminator model with source and target images as input and patch output as output
	discriminator_model = Model([source_image, target_image], patch_output)
	
	# Define optimizer for the model and compile it
	optimizer = Adam(lr=0.0002, beta_1=0.5)
	discriminator_model.compile(loss=wasserstein_loss, optimizer=optimizer, loss_weights=[0.5])    
	
	return discriminator_model


def encoder_block(input_layer, num_filters, use_batchnorm=True):
	# Initialize the kernel weights for the convolutional layer
	weight_init = RandomNormal(stddev=0.02)
	
	# Apply a 2D convolutional layer with the given number of filters and kernel size
	conv_layer = Conv2D(num_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(input_layer)
	
	# If batch normalization is desired, add a batch normalization layer
	if use_batchnorm:
		conv_layer = BatchNormalization()(conv_layer, training=True)
	
	# Apply a LeakyReLU activation function with a negative slope of 0.2
	activation_layer = LeakyReLU(alpha=0.2)(conv_layer)
	
	return activation_layer


def decoder_block(input_layer, skip_layer, num_filters, use_dropout=True):
	# Initialize the kernel weights for the transposed convolutional layer
	weight_init = RandomNormal(stddev=0.02)
	
	# Apply a 2D transposed convolutional layer with the given number of filters and kernel size
	transpose_conv_layer = Conv2DTranspose(num_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(input_layer)
	
	# Add batch normalization to the transposed convolutional layer
	transpose_conv_layer = BatchNormalization()(transpose_conv_layer, training=True)
	
	# If dropout is desired, add a dropout layer to the output of the transposed convolutional layer
	if use_dropout:
		transpose_conv_layer = Dropout(0.5)(transpose_conv_layer, training=True)
	
	# Concatenate the output of the transposed convolutional layer with the skip connection
	concatenation_layer = Concatenate()([transpose_conv_layer, skip_layer])
	
	# Apply a ReLU activation function to the concatenated layer
	activation_layer = Activation('relu')(concatenation_layer)
	
	return activation_layer



# Generator
def generator(image_shape=(256,256,3)):
	# Weight initialization function
	weight_init = RandomNormal(stddev=0.02)

	# Input image
	input_image = Input(shape=image_shape)
	
	# Encoder blocks
	num_blocks = 7
	encoders = [input_image]
	for i in range(num_blocks):
		num_filters = 64 * (2**i)
		if num_filters == 64:
			encoders.append(encoder_block(encoders[-1], num_filters, use_batchnorm=False))
		else:
			encoders.append(encoder_block(encoders[-1], num_filters))
	
	# Bottleneck block
	bottleneck = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(encoders[-1])
	bottleneck = Activation('relu')(bottleneck)
	
	# Decoder blocks
	decoders = [bottleneck]
	j = 0
	for i in range(num_blocks):		
		if i < 3:
			decoders.append(decoder_block(decoders[-1], encoders[num_blocks-i], 512))
		else:	
			num_filters = 512 // (2**j)	
			decoders.append(decoder_block(decoders[-1], encoders[num_blocks-i], num_filters, use_dropout=False))
			j += 1



	# Output image
	output_image = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(decoders[-1])
	output_image = Activation('tanh')(output_image)

	# Create the model
	model = Model(input_image, output_image)
	return model

#  combined generator and discriminator model
def GAN(generator_model, discriminator_model, image_shape):
	# Freeze the layers in the discriminator that are not BatchNormalization layers
    for layer in discriminator_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
	
	# Create an input layer for the GAN
    input_layer = Input(shape=image_shape)
    
	# Use the generator model to generate an image from the input layer
    generated_image = generator_model(input_layer)
    
	# Use the discriminator model to classify the generated image and the input layer
    discriminator_output = discriminator_model([input_layer, generated_image])
    
	# Define the GAN model that takes the input layer and outputs the discriminator's classification of the generated image and the generated image itself
    gan_model = Model(input_layer, [discriminator_output, generated_image])
    
	# Define the optimizer to use for training the GAN
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    
    # Compile the GAN model with binary cross-entropy loss for the discriminator output and mean absolute error for the generated image, using the defined optimizer
    gan_model.compile(loss=[wasserstein_loss, 'mae'], optimizer=optimizer, metrics=['accuracy'], loss_weights=[1,100])
    
    # Return the compiled GAN model
    return gan_model


def load_real_images(filename):
    # Load the data from the given filename
    data = load(filename)
    
    # Extract image data from arrays in data
    X_realA, X_realB = data['arr_0'], data['arr_1']
    
	# Normalize the pixel values of each array to [-1, 1]
    X_realA = (X_realA - 127.5) / 127.5
    X_realB = (X_realB - 127.5) / 127.5
    
    # Return preprocessed real image data
    return [X_realA, X_realB]


def generate_real_images(dataset, n_samples, patch_shape):
    # Unpack preprocessed real image data from dataset
    X_realA, X_realB = dataset
    
    # Select random samples from dataset
    ix = randint(0, X_realA.shape[0], n_samples)
    X_realA_samples, X_realB_samples = X_realA[ix], X_realB[ix]
    
    # Generate labels for real samples
    y = ones((n_samples, patch_shape, patch_shape, 1))
    
    # Return real image samples and labels
    return [X_realA_samples, X_realB_samples], y


def generate_fake_images(generator_model, samples, patch_shape):
    # Use generator model to generate fake samples
    X_fake = generator_model.predict(samples)
    
    # Generate labels for fake samples
    y = zeros((len(X_fake), patch_shape, patch_shape, 1))
    
    # Return fake image samples and labels
    return X_fake, y


def stats(step, generator_model, dataset, n_images=3):
	# Generate real and fake images
	[real_images_A, real_images_B], _ = generate_real_images(dataset, n_images, 1)
	fake_images_B, _ = generate_fake_images(generator_model, real_images_A, 1)
	
	# Scale the pixel values from [-1, 1] to [0, 1]
	real_images_A = (real_images_A + 1) / 2.0
	real_images_B = (real_images_B + 1) / 2.0
	fake_images_B = (fake_images_B + 1) / 2.0

	# Plot and save the images
	for i in range(n_images):
		# Plot real source images
		pyplot.subplot(3, n_images, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(real_images_A[i])
	
	# Plot generated target image
	for i in range(n_images):
		pyplot.subplot(3, n_images, 1 + n_images + i)
		pyplot.axis('off')
		pyplot.imshow(fake_images_B[i])
	
	# Plot real target image
	for i in range(n_images):
		pyplot.subplot(3, n_images, 1 + n_images*2 + i)
		pyplot.axis('off')
		pyplot.imshow(real_images_B[i])

	# Save the plot
	plot_filename = 'plot_%06d.png' % (step+1)
	pyplot.savefig(plot_filename)
	pyplot.close()

	# save the generator model
	model_filename = 'model_%06d.h5' % (step+1)
	generator_model.save(model_filename)
	print('>Saved: %s and %s' % (plot_filename, model_filename))

# training
def train(discriminator_model, generator_model, gan_model, dataset, num_epochs=100, batch_size=1):
	# determine the size of the discriminator's output patch
	patch_size = discriminator_model.output_shape[1]
	
	# unpack the dataset
	source_images, target_images = dataset
	
	# calculate the number of batches per epoch
	batches_per_epoch = int(len(source_images) / batch_size)
	
	# calculate the total number of training steps
	total_steps = batches_per_epoch * num_epochs

	# loop through each training step
	for i in range(total_steps):
		# generate a batch of real and fake samples
		[real_samples, real_labels], y_real = generate_real_images(dataset, batch_size, patch_size)
		fake_samples, fake_labels = generate_fake_images(generator_model, real_samples, patch_size)
		
		# update the discriminator model weights
		d_loss1 = discriminator_model.train_on_batch([real_samples, real_labels], y_real)
		d_loss2 = discriminator_model.train_on_batch([real_samples, fake_samples], fake_labels)
		
		# update the generator model weights via the GAN model
		g_loss, _, _, _, _ = gan_model.train_on_batch(real_samples, [y_real, real_labels])
		
		# print the losses for each step
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		
		# generate example images and save the generator model every 10 epochs
		if (i+1) % (batches_per_epoch * 10) == 0:
			stats(i, generator_model, dataset)

 
def load_data(directory_path, target_size=(256,512)):
    # initialize empty lists to store source and target images
    source_images, target_images = list(), list()
    
    # loop through all files in the directory
    for filename in listdir(directory_path):
        # load image from file
        image = load_img(directory_path + filename, target_size=target_size)
        # convert image to numpy array
        image = img_to_array(image)
        # split image into source and target images
        source_image, target_image = image[:, :256], image[:, 256:]
        # append source and target images to their respective lists
        source_images.append(source_image)
        target_images.append(target_image)
    
    # convert source and target image lists to numpy arrays and return as a list
    return [asarray(source_images), asarray(target_images)]


# set the path to the directory containing the dataset
dataset_path = 'FinalDataset/'

# load the source and target images from the dataset
[target_images, source_images] = load_data(dataset_path)

# print the shapes of the loaded images
print('Loaded: ', source_images.shape, target_images.shape)

# save the loaded images as a compressed numpy array file
file_name = 'images_numpy.npz'
savez_compressed(file_name, source_images, target_images)

# print a message indicating that the dataset has been saved
print('Saved dataset: ', file_name)

# load the dataset
data = load('images_numpy.npz')
source_images, target_images = data['arr_0'], data['arr_1']
print('Loaded: ', source_images.shape, target_images.shape)

# plot source images
num_images = 3
for i in range(num_images):
 pyplot.subplot(2, num_images, 1 + i)
 pyplot.axis('off')
 pyplot.imshow(source_images[i].astype('uint8'))

# plot target image
for i in range(num_images):
 pyplot.subplot(2, num_images, 1 + num_images + i)
 pyplot.axis('off')
 pyplot.imshow(target_images[i].astype('uint8'))
pyplot.show()

dataset = load_real_images('images_numpy.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)


start_time = time.time()
image_shape = dataset[0].shape[1:]
discriminator_model = create_discriminator(image_shape)
generator_model = generator(image_shape)
gan_model = GAN(generator_model, discriminator_model, image_shape)
train(discriminator_model, generator_model, gan_model, dataset)
end_time = time.time()
print("total training time = ", end_time - start_time)
