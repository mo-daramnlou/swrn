import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, PReLU, Add
from tensorflow.keras.models import Model

class ELSR(tf.keras.Model):
    """
    Implements the super-resolution model architecture from the diagram.

    This model uses a residual block and a final PixelShuffle layer (depth_to_space)
    to upscale low-resolution images.

    Attributes:
        upscale_factor (int): The factor by which to increase image resolution.
        channels (int): The number of channels in the input/output image.
    """

    def __init__(self, upscale_factor=4, channels=3):
        """Initializes the SuperResolutionNet model layers."""
        super().__init__()
        self.upscale_factor = upscale_factor
        self.channels = channels
        
        # The number of channels for the PixelShuffle layer's input is calculated
        # as C * r^2, where C is the number of output channels and r is the upscale factor.
        conv_channels_for_shuffle = self.channels * (self.upscale_factor ** 2)

        # --- Layer Definitions ---

        # Block 1: Initial Convolution
        self.conv1 = Conv2D(6, (3, 3), padding='same', name='conv_1')

        # Block 2: Residual Block
        self.conv2 = Conv2D(6, (3, 3), padding='same', name='res_conv_1')
        self.prelu = PReLU(shared_axes=[1, 2], name='prelu')
        self.conv3 = Conv2D(6, (3, 3), padding='same', name='res_conv_2')
        self.add = Add(name='residual_add')

        # Block 3: Pre-Upscaling Convolution
        self.conv4 = Conv2D(conv_channels_for_shuffle, (3, 3), padding='same', name='conv_before_shuffle')

    def call(self, inputs, training=False):
        """
        Defines the forward pass of the model.

        Args:
            inputs: The input tensor (low-resolution image).
            training: A boolean indicating if the model is in training mode.

        Returns:
            The output tensor (high-resolution image).
        """
        # Pass input through the first convolutional layer
        x = self.conv1(inputs)
        
        # Store the output for the skip connection
        skip_connection = x

        # Pass through the residual block
        x = self.conv2(x)
        x = self.prelu(x)
        x = self.conv3(x)

        # Add the skip connection
        x = self.add([skip_connection, x])

        # Pass through the pre-upscaling convolution
        x = self.conv4(x)

        # Perform the upscaling using PixelShuffle (depth_to_space)
        outputs = tf.nn.depth_to_space(x, self.upscale_factor)
        
        return outputs

    def build_graph(self):
        """
        Builds a Keras Model from the class for summary and saving.
        """
        # Define the input shape. 'None' allows for variable image sizes.
        inputs = Input(shape=(None, None, self.channels))
        # Create the model by passing the input tensor to the call method
        return Model(inputs=inputs, outputs=self.call(inputs))

if __name__ == '__main__':
    # Instantiate the model class
    super_res_net = ELSR(upscale_factor=4, channels=3)

    # Build the Keras model graph to be able to print the summary
    model = super_res_net.build_graph()

    # Print a summary of the model's architecture
    print("Model Architecture Summary:")
    model.summary()

    # You can also visualize the model architecture (requires pydot and graphviz)
    # tf.keras.utils.plot_model(model, to_file='model_diagram.png', show_shapes=True)
