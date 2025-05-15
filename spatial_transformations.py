# Copyright 2021 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
# and Applied Computer Vision Lab, Helmholtz Imaging Platform
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from builtins import range

import numpy as np
import torch
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug




def augment_spatial(data, seg, patch_size, patch_center_dist_from_border=30,
                    do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                    do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                    do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                    p_rot_per_axis: float = 1, p_independent_scale_per_axis: int = 1, glb_data=None):
    dim = len(patch_size)
    seg_result = None
    angles = np.zeros((data.shape[0], 3))
    if seg is not None:
        if dim == 2:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                                  dtype=np.float32)

    if dim == 2:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
    else:
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)

    if glb_data is not None:
        glb_result = np.zeros((data.shape[0], data.shape[1],
                               glb_data.shape[-3], glb_data.shape[-2], glb_data.shape[-1]),
                               dtype=np.float32)

    if not isinstance(patch_center_dist_from_border, (list, tuple, np.ndarray)):
        patch_center_dist_from_border = dim * [patch_center_dist_from_border]

    for sample_id in range(data.shape[0]):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        coords_glb = create_zero_centered_coordinate_mesh(glb_data.shape[2:])
        modified_coords = False

        # if do_elastic_deform and np.random.uniform() < p_el_per_sample:
        #     a = np.random.uniform(alpha[0], alpha[1])
        #     s = np.random.uniform(sigma[0], sigma[1])
        #     coords = elastic_deform_coordinates(coords, a, s)
        #     modified_coords = True

        if do_rotation and np.random.uniform() < p_rot_per_sample:

            if np.random.uniform() <= p_rot_per_axis:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= p_rot_per_axis:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= p_rot_per_axis:
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
                coords_glb = rotate_coords_3d(coords_glb, a_x, a_y, a_z)
                angles[sample_id, :] = np.array([a_x, a_y, a_z])
            else:
                coords = rotate_coords_2d(coords, a_x)
            modified_coords = True

        if do_scale and np.random.uniform() < p_scale_per_sample:
            if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc.append(np.random.uniform(scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
            else:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])

            coords = scale_coords(coords, sc)
            coords_glb = scale_coords(coords_glb, sc)
            modified_coords = True

        # now find a nice center location
        if modified_coords:
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(patch_center_dist_from_border[d],
                                            data.shape[d + 2] - patch_center_dist_from_border[d])
                else:
                    ctr = data.shape[d + 2] / 2. - 0.5
                    glb_ctr = glb_data.shape[d + 2] / 2. - 0.5

                coords[d] += ctr
                coords_glb[d] += glb_ctr

            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data,
                                                                     border_mode_data, cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg,
                                                                        border_mode_seg, cval=border_cval_seg,
                                                                        is_seg=True)
            if glb_data is not None:
                for channel_id in range(data.shape[1]):
                    glb_result[sample_id, channel_id] = interpolate_img(glb_data[sample_id, channel_id], coords_glb,
                                                                        order_data,
                                                                        border_mode_data, cval=border_cval_data)
        else:
            if seg is None:
                s = None
            else:
                s = seg[sample_id:sample_id + 1]
            if random_crop:
                margin = [patch_center_dist_from_border[d] - patch_size[d] // 2 for d in range(dim)]
                d, s = random_crop_aug(data[sample_id:sample_id + 1], s, patch_size, margin)
            else:
                d, s = center_crop_aug(data[sample_id:sample_id + 1], patch_size, s)
            data_result[sample_id] = d[0]
            if seg is not None:
                seg_result[sample_id] = s[0]
            if glb_data is not None:
                glb_result[sample_id] = glb_data[sample_id]
    return data_result, seg_result, glb_result, angles


if __name__ == '__main__':
    import SimpleITK as sitk
    import matplotlib.pyplot as plt
    import torch
    from dataloading import get_patch_size

    img1 = sitk.GetArrayFromImage(sitk.ReadImage(r'/home/genshi/nnUNet/nnUNet_raw_data_base/'
                                                 r'Dataset301_TubeTK/imagesTr/Normal003-MRA_0000.nii.gz')).astype(float)
    img2 = sitk.GetArrayFromImage(sitk.ReadImage(r'/home/genshi/nnUNet/nnUNet_raw_data_base/'
                                                 r'Dataset301_TubeTK/imagesTr/Normal006-MRA_0000.nii.gz')).astype(float)

    glb_img = np.concatenate([img1[np.newaxis, np.newaxis, :, :, :], img2[np.newaxis, np.newaxis, :, :, :]], 0)

    data = glb_img[:, :, :, 224-77: 224+77, 120-77: 120+77]

    # glb_img = glb_img[:, :, :, 47:400, 80:350]

    # glb_img = torch.from_numpy(glb_img).float()
    # glb_img = torch.nn.functional.interpolate(glb_img, size=(96, 160, 160), mode='trilinear').numpy()

    rotation_for_DA = {
        'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    }

    initial_patch_size = get_patch_size((96, 160, 160),
                                        *rotation_for_DA.values(),
                                        (0.85, 1.25))

    plt.imshow(glb_img[0, 0].max(0), cmap='gray')
    plt.show()

    seg = None

    data_result, seg_result, glb_result = \
                    augment_spatial(data, seg, (96, 96, 96), patch_center_dist_from_border=30, do_elastic_deform=False,
                    alpha=(0, 0), sigma=(0, 0),
                    do_rotation=True,
                    angle_x=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    angle_y=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    angle_z=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    do_scale=True, scale=(0.7, 1.4), border_mode_data='constant', border_cval_data=0, order_data=3,
                    border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=False, p_el_per_sample=1,
                    p_scale_per_sample=1, p_rot_per_sample=1, independent_scale_for_each_axis=False,
                    p_rot_per_axis=1, p_independent_scale_per_axis=1, glb_data=glb_img)




    print(data_result.shape, glb_result.shape)


    plt.imshow(data_result[0].squeeze().max(0), cmap='gray')
    plt.axis('off')
    plt.show()
    # plt.imshow(data_result[0].squeeze().max(1), cmap='gray')
    # plt.axis('off')
    # plt.show()
    # plt.imshow(data_result[0].squeeze().max(2), cmap='gray')
    # plt.axis('off')
    # plt.show()


    plt.imshow(glb_result[0].squeeze().max(0), cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(glb_result[0].squeeze()[:, 224-77: 224+77, 120-77: 120+77].max(0), cmap='gray')
    plt.axis('off')
    plt.show()


    # plt.imshow(glb_result[0].squeeze().max(1), cmap='gray')
    # plt.axis('off')
    # plt.show()
    # plt.imshow(glb_result[0].squeeze().max(2), cmap='gray')
    # plt.axis('off')
    # plt.show()
