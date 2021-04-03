import tensorflow as tf
from tensorflow.keras import regularizers, layers, Model, initializers

from .resblock import res_block


def get_network(weight_decay=1e-8):
    activation = tf.nn.elu
    kernel_initializer = initializers.TruncatedNormal(stddev=1e-3)
    regularizer = regularizers.L2(l2=weight_decay)

    inp_img = layers.Input((128, 64, 3))

    x = inp_img

    x = layers.Conv2D(32, (3, 3), 1, 'same', activation=activation,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=regularizer,
                      bias_regularizer=regularizer,
                      name='conv1_1')(x)

    x = layers.BatchNormalization(name='conv1_1_bn')(x)

    x = layers.Conv2D(32, (3, 3), 1, 'same', activation=activation,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=regularizer,
                      bias_regularizer=regularizer,
                      name='conv1_2')(x)

    x = layers.BatchNormalization(name='conv1_2_bn')(x)

    x = layers.MaxPool2D((3, 3), (2, 2), padding='same', name='pool_1')(x)

    x = res_block(
        x, "resblock_2_1", activation, kernel_initializer, regularizer, increase_dim=False, is_first=True
    )

    x = res_block(
        x, "resblock_2_3", activation, kernel_initializer, regularizer, increase_dim=False,
    )

    x = res_block(
        x, "resblock_3_1", activation, kernel_initializer, regularizer, increase_dim=True,
    )

    x = res_block(
        x, "resblock_3_3", activation, kernel_initializer, regularizer, increase_dim=False,
    )

    x = res_block(
        x, "resblock_4_1", activation, kernel_initializer, regularizer, increase_dim=True,
    )

    x = res_block(
        x, "resblock_4_3", activation, kernel_initializer, regularizer, increase_dim=False
    )

    feature_dim = x.get_shape().as_list()[-1]

    x = layers.Flatten()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(feature_dim, activation=activation,
                     kernel_regularizer=regularizer,
                     bias_regularizer=regularizer,
                     kernel_initializer=kernel_initializer,
                     name='fc_1')(x)

    x = layers.BatchNormalization(name='last_norm')(x)
    x = tf.nn.l2_normalize(x, axis=1)

    model = Model(inputs=inp_img, outputs=x)
    return model


def preprocess(image, is_training=False, input_is_bgr=False):
    if input_is_bgr:
        image = image[:, :, ::-1]
    image = tf.divide(tf.cast(image, tf.float32), 255.0)
    if is_training:
        image = tf.image.random_flip_left_right(image)
    return image


if __name__ == '__main__':
    network = get_network()
    print(network.summary())
