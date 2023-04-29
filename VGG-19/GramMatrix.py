import tensorflow as tf


class GramMatrix:
    def __call__(self, input_tensor):
        print("GramMatrix __call__")
        # Compute the Gram matrix using the einsum function
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        # Normalize the Gram matrix by the number of locations
        return tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor) / locations
