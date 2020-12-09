from .ssd300.network import get_network as get_network_300
from .ssd512.network import get_network as get_network_512
from tracker.configs import DETECTOR_CHECKPOINT_PATH


def get_network(num_classes, architecture, train=True):
    assert architecture in ['ssd300', 'ssd512']
    model = get_network_300(num_classes) if architecture == 'ssd300' else get_network_512(num_classes)

    if not train:
        print('Loading weights from {}'.format(DETECTOR_CHECKPOINT_PATH))
        model.load_weights(DETECTOR_CHECKPOINT_PATH)
        model.save('ssd512_checkpoint')
    return model
