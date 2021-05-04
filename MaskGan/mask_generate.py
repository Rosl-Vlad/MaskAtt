import torch
import numpy as np
import hickle as hkl

from torch.nn.functional import interpolate

"""
0: 'background'	1: 'skin'	2: 'l_eye'
3: 'r_eye'	4: 'l_eye'	5: 'r_eye'
6: 'eye_g'	7: 'l_ear'	8: 'r_ear'
9: 'ear_r'	10: 'nose'	11: 'mouth'
12: 'u_lip'	13: 'l_lip'	14: 'neck'
15: 'neck_l' 16: 'cloth' 17: 'hair'
18: 'hat'
"""

mask_influence = np.array(
    [
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Bangs
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Black_Hair
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Blond_Hair
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Brown_Hair
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Bushy_Eyebrows
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Eyeglasses
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],  # Male
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # Mouth_Slightly_Open
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Mustache
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # No_Beard
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # Pale_Skin
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0],  # Young
    ]
)


def get_influence_mask(mask_path, attr, size=(128, 128)):
    mask = hkl.load(mask_path)
    res = np.zeros((len(attr), mask.shape[0], mask.shape[1]))
    attr = abs(attr.astype(bool))
    v = np.argwhere(mask_influence[attr] == 1)
    influence_mask_classes_per_attr = np.split(v[:, 1], (np.argwhere(np.diff(v, axis=0)[:, 0] != 0) + 1).flatten())
    for i, arg_attr in enumerate(np.argwhere(attr).flatten()):
        mask_c = mask.copy()
        mask_c[np.isin(mask_c, influence_mask_classes_per_attr[i]) == False] = 0
        mask_c[np.isin(mask_c, influence_mask_classes_per_attr[i])] = 1
        res[arg_attr] = mask_c
        res[arg_attr] = interpolate(torch.tensor(np.expand_dims(mask_c, axis=(0, 1))), size=size)[0][0].numpy()
    return res


mask_influence_v2 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Bangs
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Black_Hair
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Blond_Hair
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Brown_Hair
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Bushy_Eyebrows
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Eyeglasses
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],  # Male
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # Mouth_Slightly_Open
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Mustache
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # No_Beard
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # Pale_Skin
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0],  # Young
    ]
)


def get_influence_mask_v2(mask_path, attr, size=(128, 128)):
    mask = hkl.load(mask_path)
    res = np.zeros((len(attr), mask.shape[0], mask.shape[1]))
    attr = abs(attr.astype(bool))
    v = np.argwhere(mask_influence_v2[np.ones_like(attr)] == 1)
    influence_mask_classes_per_attr = np.split(v[:, 1], (np.argwhere(np.diff(v, axis=0)[:, 0] != 0) + 1).flatten())
    print(influence_mask_classes_per_attr)
    for i, arg_attr in enumerate(range(len(attr))):
        mask_c = mask.copy().astype(float)
        mask_c[np.isin(mask_c, influence_mask_classes_per_attr[i]) == False] = 0
        mask_c[np.isin(mask_c, influence_mask_classes_per_attr[i])] *= 1/19
        res[arg_attr] = mask_c
        res[arg_attr] = interpolate(torch.tensor(np.expand_dims(mask_c, axis=(0, 1))), size=size)[0][0].numpy()
    return res
