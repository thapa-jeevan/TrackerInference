from tensorflow.python.keras import layers, initializers, models

from .backbone import get_backbone
from tracker.detector.configs.dataset import NO_CLASSES

initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.03)


def get_detection_head(num_classes):
    classification_head_layers = [
        layers.Conv2D(6 * num_classes, 3, padding='same', use_bias=False, kernel_initializer=initializer,
                      name='classifier1'),
        layers.Conv2D(6 * num_classes, 3, padding='same', use_bias=False, kernel_initializer=initializer,
                      name='classifier2'),
        layers.Conv2D(6 * num_classes, 3, padding='same', use_bias=False, kernel_initializer=initializer,
                      name='classifier3'),
        layers.Conv2D(4 * num_classes, 3, padding='same', use_bias=False, kernel_initializer=initializer,
                      name='classifier4'),
        layers.Conv2D(4 * num_classes, 1, use_bias=False, kernel_initializer=initializer, name='classifier5')
    ]

    localization_head_layers = [
        layers.Conv2D(6 * 4, 3, padding='same', use_bias=False, kernel_initializer=initializer, name='localizer1'),
        layers.Conv2D(6 * 4, 3, padding='same', use_bias=False, kernel_initializer=initializer, name='localizer2'),
        layers.Conv2D(6 * 4, 3, padding='same', use_bias=False, kernel_initializer=initializer, name='localizer3'),
        layers.Conv2D(4 * 4, 3, padding='same', use_bias=False, kernel_initializer=initializer, name='localizer4'),
        layers.Conv2D(4 * 4, 1, use_bias=False, kernel_initializer=initializer, name='localizer5')
    ]
    return classification_head_layers, localization_head_layers


def get_network(num_classes):
    backbone = get_backbone(alpha=1.0)
    classification_head_layers, localization_head_layers = get_detection_head(num_classes)

    img_input = layers.Input(shape=(300, 300, 3))
    x = img_input
    classification_outputs = []
    localization_outputs = []

    for idx, block in enumerate(backbone):
        x = block(x)

        _classification_output = classification_head_layers[idx](x)
        _localization_output = localization_head_layers[idx](x)

        _classification_output = layers.Reshape(target_shape=(-1, num_classes),
                                                name='reshape_classification_' + str(idx))(_classification_output)
        _localization_output = layers.Reshape(target_shape=(-1, 4),
                                              name='reshape_localization_' + str(idx))(_localization_output)

        classification_outputs.append(_classification_output)
        localization_outputs.append(_localization_output)

    classification_outputs_ = layers.Concatenate(axis=1)(classification_outputs)
    localization_outputs_ = layers.Concatenate(axis=1)(localization_outputs)

    model = models.Model(inputs=img_input, outputs=[classification_outputs_, localization_outputs_])
    print('Using SSD 300 network with {} classes including Background'.format(num_classes))

    return model


if __name__ == '__main__':
    model = get_network(NO_CLASSES)
    model.summary()
