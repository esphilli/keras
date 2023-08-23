import tensorflow.compat.v2 as tf
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export

BASE_WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/densenet/"
)

DENSENET121_WEIGHT_PATH_NO_TOP = (
    BASE_WEIGHTS_PATH
    + "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

layers = VersionAwareLayers()


def dense_block(x, blocks, name):
    """A dense block.
    Args:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.
    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + "_block" + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    Args:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.
    Returns:
      output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x = layers.GroupNormalization(
        groups=2, axis=bn_axis, epsilon=1.001e-5, name=name + "_gn"
    )(x)
    x = layers.Activation("relu", name=name + "_relu")(x)
    x = layers.Conv3D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
        name=name + "_conv",
    )(x)
    x = layers.AveragePooling3D(2, strides=2, name=name + "_pool")(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    Args:
      x: input tensor.
      growth_rate: float, growth rate at dense layers.
      name: string, block label.
    Returns:
      Output tensor for the block.
    """
    bn_axis = 4 if backend.image_data_format() == "channels_last" else 1
    x1 = layers.GroupNormalization(
        groups=2, axis=bn_axis, epsilon=1.001e-5, name=name + "_0_gn"
    )(x)
    x1 = layers.Activation("relu", name=name + "_0_relu")(x1)
    x1 = layers.Conv3D(
        4 * growth_rate, 1, use_bias=False, name=name + "_1_conv"
    )(x1)
    x1 = layers.GroupNormalization(
        groups=2, axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn"
    )(x1)
    x1 = layers.Activation("relu", name=name + "_1_relu")(x1)
    x1 = layers.Conv3D(
        growth_rate, 3, padding="same", use_bias=False, name=name + "_2_conv"
    )(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])
    return x


def DenseNet(
    blocks,
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
   
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded."
        )

    # Determine proper input shape
    #input_shape = imagenet_utils.obtain_input_shape(
    #    input_shape,
    #    default_size=224,
    #    min_size=32,
    #    data_format=backend.image_data_format(),
    #    require_flatten=include_top,
    #    weights=weights,
    #)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 4 if backend.image_data_format() == "channels_last" else 1

    x = layers.ZeroPadding3D(padding=(3, 3, 3))(img_input)
    x = layers.Conv3D(64, 7, strides=2, use_bias=False, name="conv1/conv")(x)
    x = layers.GroupNormalization(
        groups=2, axis=bn_axis, epsilon=1.001e-5, name="conv1/gn"
    )(x)
    x = layers.Activation("relu", name="conv1/relu")(x)
    x = layers.ZeroPadding3D(padding=(1, 1, 1))(x)
    x = layers.MaxPooling3D(3, strides=2, name="pool1")(x)

    x = dense_block(x, blocks[0], name="conv2")
    x = transition_block(x, 0.5, name="pool2")
    x = dense_block(x, blocks[1], name="conv3")
    x = transition_block(x, 0.5, name="pool3")
    x = dense_block(x, blocks[2], name="conv4")
    x = transition_block(x, 0.5, name="pool4")
    x = dense_block(x, blocks[3], name="conv5")

    x = layers.GroupNormalization(groups=2, axis=bn_axis, epsilon=1.001e-5, name="gn")(x)
    x = layers.Activation("relu", name="relu")(x)

    if pooling == "avg":
        x = layers.GlobalAveragePooling3D(name="avg_pool")(x)
    elif pooling == "max":
        x = layers.GlobalMaxPooling3D(name="max_pool")(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = training.Model(inputs, x, name="densenet121")
    else:
        model = training.Model(inputs, x, name="densenet")

    # Load weights.
    if weights == "imagenet":
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = data_utils.get_file(
                    "densenet121_weights_tf_dim_ordering_tf_kernels.h5",
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir="models",
                    file_hash="9d60b8095a5708f2dcce2bca79d332c7",
                )           
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = data_utils.get_file(
                    "densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir="models",
                    file_hash="30ee3e1110167f948a6b9946edeeb738",
                )
            
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


@keras_export(
    "keras.applications.densenet.DenseNet121", "keras.applications.DenseNet121"
)
def DenseNet121(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
):
    """Instantiates the Densenet121 architecture."""
    return DenseNet(
        [6, 12, 24, 16],
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation,
    )


@keras_export("keras.applications.densenet.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="torch"
    )

@keras_export("keras.applications.densenet.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)

preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_TORCH,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__

#setattr(DenseNet121, "__doc__", DenseNet121.__doc__ + DOC)
