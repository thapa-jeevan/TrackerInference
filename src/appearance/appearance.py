import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as transforms

PATCH_SIZE = (128, 64, 3)
PATCH_BATCH_SIZE = 4
CHECKPOINT_APPEARANCE = 'checkpoints/appearance/resnet-10-checkpoint'


def get_model():
    model_appearance = torch.load(CHECKPOINT_APPEARANCE)
    data_transforms = transforms.Compose([
        transforms.Resize(PATCH_SIZE[:-1]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def encode(img, boxes):
        patch_batches = preprocess(img, boxes, data_transforms)
        features = []
        for inp_patch in patch_batches:
            embed = model_appearance(inp_patch.cuda())
            features.append(embed)
        if len(features) > 0:
            features = torch.cat(features, dim=0).detach().cpu()
        return features

    return encode


def preprocess(img, boxes, data_transforms):
    patches = []
    for box in boxes:
        x1, y1, x2, y2 = box.int()
        patch = img[y1:y2, x1:x2]
        patch = Image.fromarray(np.uint8(patch))
        patches.append(data_transforms(patch))
    if len(patches) > 0:
        patch_batches = torch.stack(patches).split(PATCH_BATCH_SIZE)
    else:
        patch_batches = []
    return patch_batches
