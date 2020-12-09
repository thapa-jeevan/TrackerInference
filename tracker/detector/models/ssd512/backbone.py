import os

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers, initializers
from tensorflow.python.keras.engine import training

from tracker.configs.settings import WEIGHTS_DIR

initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.03)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, block_partition=0):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if not block_partition == 2:
        if block_id:
            x = layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False,
                              name=prefix + 'expand')(x)
            x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
            x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
        else:
            prefix = 'expanded_conv_'

    if block_partition == 1:
        return x

    if stride == 2:
        x = layers.ZeroPadding2D(padding=1, name=prefix + 'pad')(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, use_bias=False,
                               padding='same' if stride == 1 else 'valid', name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    x = layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])

    return x


def conv_block(inputs, stride, filters, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    x = inputs
    prefix = 'attached_conv_block_{}_'.format(block_id)
    x = layers.Conv2D(int(filters / 2), 1, 1, use_bias=False, kernel_initializer=initializer, name=prefix + 'conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'BN1')(x)
    x = layers.ReLU(6., name=prefix + 'relu1')(x)

    x = layers.Conv2D(filters, 3, stride, padding='same', use_bias=False, kernel_initializer=initializer,
                      name=prefix + 'conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'BN2')(x)
    x = layers.ReLU(6., name=prefix + 'relu2')(x)
    return x


def get_feature_extractor(alpha=1.0):
    img_input = layers.Input(shape=(512, 512, 3))
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.ZeroPadding2D(padding=1, name='Conv1_pad')(img_input)
    x = layers.Conv2D(first_block_filters, kernel_size=3, strides=(2, 2), padding='valid', use_bias=False,
                      name='Conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13, block_partition=1)
    model1 = training.Model(img_input, x, name='SSD_block_1')
    weight_path = os.path.join(WEIGHTS_DIR, 'SSD-Block1.hd5')
    model1.load_weights(weight_path)

    img_input = layers.Input(shape=(32, 32, 96 * 6))

    x = img_input
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13, block_partition=2)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)
    model2 = training.Model(img_input, x, name='SSD_block_2')
    weight_path = os.path.join(WEIGHTS_DIR, 'SSD-Block2.hd5')
    model2.load_weights(weight_path)

    return model1, model2


def get_following_layers(alpha=1.0):
    img_input = layers.Input((16, 16, 320))
    x = img_input
    x = conv_block(x, stride=2, filters=256, block_id=1)
    model3 = training.Model(img_input, x, name='SSD_block_3')

    img_input = layers.Input((8, 8, 256))
    x = img_input
    x = conv_block(x, stride=2, filters=256, block_id=2)
    model4 = training.Model(img_input, x, name='Block4')

    img_input = layers.Input((4, 4, 256))
    x = img_input
    x = conv_block(x, stride=2, filters=256, block_id=3)
    model5 = training.Model(img_input, x, name='Block5')

    img_input = layers.Input((2, 2, 256))
    x = img_input
    x = conv_block(x, stride=2, filters=128, block_id=4)
    model6 = training.Model(img_input, x, name='Block6')

    return model3, model4, model5, model6


def get_backbone(alpha=1.0):
    feature_extractor = get_feature_extractor(alpha)
    following_layers = get_following_layers(alpha)
    return feature_extractor + following_layers


if __name__ == '__main__':
    model = get_backbone(7)
    print(model.summary())
