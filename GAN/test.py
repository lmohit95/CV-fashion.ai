# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from numpy import expand_dims
from matplotlib import pyplot
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
import os

model = load_model('cgan_model_032300.h5')

def preprocess_images(path_to_image, target_size=(256, 512)):
    """
    Load and preprocess an image for feeding into a machine learning model.

    Args:
        path_to_image (str): The path to the input image file.
        target_size (tuple): The desired size to which the image is to be resized.

    Returns:
        numpy.ndarray: A preprocessed image ready to be fed into a machine learning model.
    """
    # Load and resize the image
    image = load_img(path_to_image, target_size=target_size)
    # Convert to numpy array
    image = img_to_array(image)
    # Split the image into a satellite image and a map image
    satellite_image, map_image = image[:, :256], image[:, 256:]
    # Normalize the pixel values of the map image between -1 and 1
    map_image = (map_image - 127.5) / 127.5
    # Add an extra dimension to the map image to represent the batch size of 1
    map_image = expand_dims(map_image, 0)
    return map_image


def preprocess_images_256(path_to_image, target_size=(256, 256)):
    """
    Load and preprocess an image for feeding into a machine learning model.

    Args:
        path_to_image (str): The path to the input image file.
        target_size (tuple): The desired size to which the image is to be resized.

    Returns:
        numpy.ndarray: A preprocessed image ready to be fed into a machine learning model.
    """
    # Load and resize the image
    image = load_img(path_to_image, target_size=target_size)
    # Convert to numpy array
    image = img_to_array(image)
    # Extract the map image
    map_image = image[:, :]
    # Normalize the pixel values of the map image between -1 and 1
    map_image = (map_image - 127.5) / 127.5
    # Add an extra dimension to the map image to represent the batch size of 1
    map_image = expand_dims(map_image, 0)
    return map_image


# Set the path to the input images directory
input_path = "Test_256/"
# Initialize the counter for the output image filenames
counter = 0
# Get a list of all the input image filenames
input_filenames = os.listdir(input_path)
# Loop over all the input images
for input_filename in input_filenames:
    # Construct the full path to the input image
    input_pathname = os.path.join(input_path, input_filename)
    # Preprocess the input image
    input_image = preprocess_images_256(input_pathname)
    # Use the machine learning model to generate an output image
    output_image = model.predict(input_image)
    # Rescale the pixel values of the output image from [-1, 1] to [0, 1]
    output_image = (output_image + 1) / 2.0
    # Plot and save the output image
    pyplot.imshow(output_image[0])
    pyplot.axis('off')
    output_pathname = 'result/result_' + str(counter) + input_filename
    pyplot.savefig(output_pathname)
    pyplot.show()
    # Increment the counter for the output image filenames
    counter += 1