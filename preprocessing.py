import pickle
import numpy as np
from typing import Union, Tuple, List
import SimpleITK as sitk
from skimage.morphology import remove_small_objects
import os
import tqdm
import torch
import pandas as pd
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize
from collections import OrderedDict
import json



def generate_mask(img_3D, axis, method='mean', thread=0.65):
    assert method in ['mean', 'mip', 'std', 'grad']
    assert axis in [0, 1, 2]
    if method == 'mean':
        mask = img_3D.mean(axis)
    elif method == 'mip':
        mask = img_3D.max(axis)
    elif method == 'std':
        mask = img_3D.std(axis)
    else:
        mask = np.gradient(img_3D)[axis].std(axis)

    if method in ['mean', 'mip', 'std']:
        mask[mask < np.quantile(mask, thread)] = 0
        mask[mask != 0] = 1
    else:
        mask[mask < mask.mean()] = 0
        mask[mask != 0] = 1

    return mask


def generate_square(img_3D, axis=0, thread=0.65):
    methods = ['mean', 'mip', 'std']
    masks = [generate_mask(img_3D, axis, method, thread) for method in methods]
    grad_mask = generate_mask(img_3D, axis, 'grad', thread)

    final_mask = np.bitwise_or(masks[0].astype(int),
                               masks[1].astype(int),
                               masks[2].astype(int))

    final_mask = np.bitwise_or(grad_mask.astype(int), final_mask)

    final_mask = remove_small_objects(final_mask.astype(bool),
                                      min_size=200, connectivity=1)

    margin_x = np.where(final_mask.sum(0) != 0)[0]
    margin_x.sort()
    margin_x0, margin_x1 = max(margin_x[0] - 1, 0), min(margin_x[-1] + 1,
                                                        final_mask.shape[1] - 1)

    margin_y = np.where(final_mask.sum(1) != 0)[0]
    margin_y.sort()
    margin_y0, margin_y1 = max(margin_y[0] - 1, 0), min(margin_y[-1] + 1,
                                                        final_mask.shape[0] - 1)
    return margin_x0, margin_x1 - 1, margin_y0, margin_y1 - 1


def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape


def get_do_separate_z(spacing: Union[Tuple[float, ...], List[float], np.ndarray], anisotropy_threshold=3):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z


def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis


def resample_data_or_seg(data: np.ndarray, new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
                         is_seg: bool = False, axis: Union[None, int] = None, order: int = 3,
                         do_separate_z: bool = False, order_z: int = 0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        data = data.astype(float)
        if do_separate_z:
            # print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,
                                                                   mode='nearest')[None])
                    else:
                        unique_labels = np.sort(pd.unique(reshaped_data.ravel()))  # np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            # print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        # print("no resampling necessary")
        return data


def resample_data_or_seg_to_shape(data: Union[torch.Tensor, np.ndarray],
                                  new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                  current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                  is_seg: bool = False,
                                  order: int = 3, order_z: int = 0,
                                  force_separate_z: Union[bool, None] = False,
                                  separate_z_anisotropy_threshold: float = 3):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the same spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


class Preprocessor(object):
    def __init__(self, img_dir, seg_dir, train_sub_list, total_sub_list, tar_dir, img_modality='MRI'):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.sub_list = total_sub_list
        self.train_subs = train_sub_list
        self.tar_dir = tar_dir
        self.img_modality = img_modality
        assert self.img_modality in ['MRI', 'CT']

        # For CT data
        self.num_foreground_voxels_for_intensitystats = 10e7
        self.num_foreground_samples_per_case = int(self.num_foreground_voxels_for_intensitystats //
                                                   len(self.train_subs))
        self.foreground_intensities_per_channel = {}
        self.train_idx = [self.sub_list.index(x) for x in self.train_subs]

    def compute_bounding_box(self, img):

        non_zero_indices = np.transpose(np.nonzero(img))
        min_coords = np.min(non_zero_indices, axis=0)
        max_coords = np.max(non_zero_indices, axis=0)

        if self.img_modality != 'CT':
            min_x, max_x, min_y, max_y = generate_square(img, axis=0, thread=0.65)
            min_coords[1], max_coords[1] = min_y, max_y
            min_coords[2], max_coords[2] = min_x, max_x

        return min_coords, max_coords

    def sample_foreground_locations(self, seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                    seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
        return class_locs

    def normalize(self, image):
        mean = image.mean()
        std = image.std()
        image = (image - mean) / (max(std, 1e-8))
        return image

    def compute_foreground_intensities(self, img, seg, seed=1234):
        # c Z Y X 4d array
        rs = np.random.RandomState(seed)
        foreground_mask = seg[0] > 0

        intensities_per_channel = {}

        for c in range(img.shape[0]):
            foreground_pixels = img[c][foreground_mask]
            num_fg = len(foreground_pixels)
            intensities_per_channel[c] = (rs.choice(foreground_pixels,
                                                    self.num_foreground_samples_per_case, replace=True)
                                          if num_fg > 0 else [])

        return intensities_per_channel

    def compute_dataset_foreground_intensities(self, img_list, seg_list, seed=1234):
        foreground_intensities_per_channel = {}

        all_intensities_per_channel = [self.compute_foreground_intensities(img, seg, seed)
                                       for (img, seg) in zip(img_list, seg_list)]
        num_c = img_list[0].shape[0]

        for c in range(num_c):
            cur_c_intensities = np.concatenate([all_intensities_per_channel[i][c] for i in range(len(img_list))])
            foreground_intensities_per_channel[c] = {
                'mean': float(np.mean(cur_c_intensities)),
                'std': float(np.std(cur_c_intensities)),
                'percentile_99_5': float(np.percentile(cur_c_intensities, 99.5)),
                'percentile_00_5': float(np.percentile(cur_c_intensities, 0.5)),
            }
        print(foreground_intensities_per_channel)
        return foreground_intensities_per_channel

    # def post_normalization_for_CT(self, img_list, seg_list, seed=1234):
    #     norm_img_list = []
    #     train_imgs, train_segs = [img_list[i] for i in self.train_idx], [seg_list[i] for i in self.train_idx]
    #     self.foreground_intensities_per_channel = self.compute_dataset_foreground_intensities(train_imgs, train_segs,
    #                                                                                           seed)
    #     for img in img_list:
    #         norm_img = img.copy().astype(np.float32)
    #         for c in range(img.shape[0]):
    #             mean_intensity = self.foreground_intensities_per_channel[c]['mean']
    #             std_intensity = self.foreground_intensities_per_channel[c]['std']
    #             lower_bound = self.foreground_intensities_per_channel[c]['percentile_00_5']
    #             upper_bound = self.foreground_intensities_per_channel[c]['percentile_99_5']
    #             norm_img[c] = np.clip(norm_img[c], lower_bound, upper_bound)
    #             norm_img[c] = (norm_img[c] - mean_intensity) / max(std_intensity, 1e-8)
    #         norm_img_list.append(norm_img)
    #
    #     return norm_img_list

    def post_normalization_for_CT(self, img_dict, seg_dict, seed=1234):
        norm_img_dict = {}
        train_imgs, train_segs = [img_dict[sub] for sub in self.train_subs], [seg_dict[sub] for sub in self.train_subs]
        self.foreground_intensities_per_channel = self.compute_dataset_foreground_intensities(train_imgs, train_segs,
                                                                                              seed)
        for sub in self.sub_list:
            img = img_dict[sub]
            norm_img = img.copy().astype(np.float32)
            for c in range(img.shape[0]):
                mean_intensity = self.foreground_intensities_per_channel[c]['mean']
                std_intensity = self.foreground_intensities_per_channel[c]['std']
                lower_bound = self.foreground_intensities_per_channel[c]['percentile_00_5']
                upper_bound = self.foreground_intensities_per_channel[c]['percentile_99_5']
                norm_img[c] = np.clip(norm_img[c], lower_bound, upper_bound)
                norm_img[c] = (norm_img[c] - mean_intensity) / max(std_intensity, 1e-8)
            norm_img_dict[sub] = norm_img

        return norm_img_dict

    def normalization_for_nonCT(self, img_dict):
        for sub in self.sub_list:
            img_dict[sub] = self.normalize(img_dict[sub])
        return img_dict


    def determine_fullres_target_spacing(self, spacings, sizes) -> np.ndarray:

        target = np.percentile(np.vstack(spacings), 50, 0)

        target_size = np.percentile(np.vstack(sizes), 50, 0)
        # we need to identify datasets for which a different target spacing could be beneficial. These datasets have
        # the following properties:
        # - one axis which much lower resolution than the others
        # - the lowres axis has much less voxels than the others
        # - (the size in mm of the lowres axis is also reduced)
        worst_spacing_axis = np.argmax(target)
        other_axes = [i for i in range(len(target)) if i != worst_spacing_axis]
        other_spacings = [target[i] for i in other_axes]
        other_sizes = [target_size[i] for i in other_axes]

        has_aniso_spacing = target[worst_spacing_axis] > (3 * max(other_spacings))
        has_aniso_voxels = target_size[worst_spacing_axis] * 3 < min(other_sizes)
        do_separate_z = False
        if has_aniso_spacing and has_aniso_voxels:
            do_separate_z = True
            spacings_of_that_axis = np.vstack(spacings)[:, worst_spacing_axis]
            target_spacing_of_that_axis = np.percentile(spacings_of_that_axis, 10)
            # don't let the spacing of that axis get higher than the other axes
            if target_spacing_of_that_axis < max(other_spacings):
                target_spacing_of_that_axis = max(max(other_spacings), target_spacing_of_that_axis) + 1e-5
            target[worst_spacing_axis] = target_spacing_of_that_axis
        return target, do_separate_z, worst_spacing_axis

    # def load_case(self, img_dir, seg_dir, sub):
    #     # assert sub end with .nii.gz
    #
    #     # load image and seg
    #     img_path = os.path.join(img_dir, sub)
    #     seg_path = os.path.join(seg_dir, sub)
    #
    # image = sitk.ReadImage(img_path)
    # spacing = np.array(image.GetSpacing())
    # img = sitk.GetArrayFromImage(image).astype(float)
    # seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(int)
    #     ori_shape = img.shape
    #
    #     # Cropping
    #     min_coords, max_coords = self.compute_bounding_box(img)
    #
    #     img = img[min_coords[0]:max_coords[0] + 1, min_coords[1]:max_coords[1] + 1,
    #           min_coords[2]:max_coords[2] + 1]
    #     seg = seg[min_coords[0]:max_coords[0] + 1, min_coords[1]:max_coords[1] + 1,
    #           min_coords[2]:max_coords[2] + 1]
    #
    #     # Normlization
    #     if self.img_modality != 'CT':
    #         img = self.normalize(img)[np.newaxis, :, :, :]
    #     else:
    #         # CT normalization will be performed after reading all training imgs
    #         img = img[np.newaxis, :, :, :]
    #
    #     # sample_foreground_locations
    #     seg = seg[np.newaxis, :, :, :]
    #     class_locs = self.sample_foreground_locations(seg, classes_or_regions=[1])
    #
    #     data_dict = {
    #         'img': img,
    #         'seg': seg,
    #         'ori_shape': ori_shape,
    #         'spacing': spacing,
    #         'class_locs': class_locs
    #     }
    #
    #     coord_dict = {
    #         'min_coords': min_coords,
    #         'max_coords': max_coords
    #     }
    #
    #     return data_dict, coord_dict

    def load_case(self, img_dir, seg_dir, sub):
        # assert sub end with .nii.gz

        # load image and seg
        img_path = os.path.join(img_dir, sub)
        seg_path = os.path.join(seg_dir, sub)

        image = sitk.ReadImage(img_path)
        spacing = np.array(image.GetSpacing())
        spacing = np.array([spacing[2], spacing[1], spacing[0]])
        img = sitk.GetArrayFromImage(image)
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).astype(int)

        return img, seg, spacing

    def run(self):
        all_data_items = {}
        # img_list, seg_list, spacing_list = [], [], []
        # min_coords_list, max_coords_list = [], []
        # ori_shps, shps_before_resample = [], []

        img_dict, seg_dict, spacing_dict = {}, {}, {}
        min_coords_dict, max_coords_dict = {}, {}
        ori_shps_dict, shps_before_resample = {}, {}

        # load training data
        print('start to load data')
        for sub in tqdm.tqdm(self.sub_list):
            img, seg, spacing = self.load_case(self.img_dir, self.seg_dir, sub)
            shp = img.shape
            ori_shps_dict[sub] = np.array(shp)

            # crop the data
            min_coords, max_coords = self.compute_bounding_box(img)
            img = img[min_coords[0]:max_coords[0] + 1, min_coords[1]:max_coords[1] + 1,
                      min_coords[2]:max_coords[2] + 1]
            seg = seg[min_coords[0]:max_coords[0] + 1, min_coords[1]:max_coords[1] + 1,
                      min_coords[2]:max_coords[2] + 1]

            min_coords_dict[sub] = min_coords
            max_coords_dict[sub] = max_coords

            shps_before_resample[sub] = np.array(img.shape)
            img_dict[sub] = img
            seg_dict[sub] = seg
            spacing_dict[sub] = spacing

        print('start to resample data')
        # resample the data
        train_spacings = [spacing_dict[sub] for sub in self.train_subs]
        train_shps_before_resample = [shps_before_resample[sub] for sub in self.train_subs]
        target_spacing, do_separate_z, axis = self.determine_fullres_target_spacing(train_spacings,
                                                                                    train_shps_before_resample)
        print(target_spacing)
        # target_spacing = np.array([target_spacing[2], target_spacing[1], target_spacing[0]])

        for sub in tqdm.tqdm(self.sub_list):

            img, seg, old_spacing = img_dict[sub], seg_dict[sub], spacing_dict[sub]

            # print(old_spacing, spacing_dict[sub])

            img, seg = img[np.newaxis, :, :, :], seg[np.newaxis, :, :, :]
            old_shp = shps_before_resample[sub]

            new_shape = compute_new_shape(old_shape=old_shp, old_spacing=old_spacing, new_spacing=target_spacing)

            resample_img = resample_data_or_seg_to_shape(data=img, new_shape=new_shape, current_spacing=old_spacing,
                                                         new_spacing=target_spacing, is_seg=False, order=3)
            print(sub, img.shape, resample_img.shape)
            if sub in self.train_subs:
                resample_seg = resample_data_or_seg_to_shape(data=seg, new_shape=new_shape, current_spacing=old_spacing,
                                                             new_spacing=target_spacing, is_seg=True, order=1)
            else:
                # do not resample the seg data for testing subs
                resample_seg = seg

            img_dict[sub] = resample_img
            seg_dict[sub] = resample_seg

        print('start to normalize data')
        # normalization
        if self.img_modality == 'CT':
            img_dict = self.post_normalization_for_CT(img_dict, seg_dict)
        else:
            img_dict = self.normalization_for_nonCT(img_dict)
        #
        #
        print('start to save data')
        # save data .npz format
        for sub in tqdm.tqdm(self.sub_list):
            tar_data_path = os.path.join(self.tar_dir, sub.replace('.nii.gz', '.npz'))
            tar_coord_path = os.path.join(self.tar_dir, sub.replace('.nii.gz', '_coord.npz'))

            data_dict = {
                'img': img_dict[sub],
                'seg': seg_dict[sub],
                'ori_shape': ori_shps_dict[sub],
                'shp_before_resample': shps_before_resample[sub],
                'spacing': spacing_dict[sub],
                'target_spacing': target_spacing,
                'class_locs': self.sample_foreground_locations(seg_dict[sub], classes_or_regions=[1])
            }

            coord_dict = {
                'min_coords': min_coords_dict[sub],
                'max_coords': max_coords_dict[sub]
            }


            if not os.path.exists(tar_data_path):
                np.savez(tar_data_path, **data_dict, allow_pickle=True)

            if not os.path.exists(tar_coord_path):
                np.savez(tar_coord_path, **coord_dict, allow_pickle=True)



if __name__ == '__main__':
    # 每个数据集有一个单独的dir
    # 单独的dir下面有 两个dir, images 和 masks
    # images和masks分别放图像和分割结果，并且每个样本在两个dir下面的名称一致

    dataset = 'KiPA'

    img_dir, seg_dir = r'./Data/raw_data/{}/images'.format(dataset), r'./Data/raw_data/{}/masks'.format(dataset)
    total_sub_list = os.listdir(seg_dir)
    split_file = r'./Data/raw_data/{}/splits_final.json'.format(args.dataset)
    data_dir = r'./Data/preprocessed_data/{}'.format(args.dataset)

    splits = json.load(f)
    train_subs = splits[0]['train']
    valid_subs = splits[0]['val']
    train_sub_list = train_subs


    train_sub_list = [sub + '.nii.gz' for sub in train_sub_list]

    # train_sub_list = total_sub_list
    tar_dir = r'./Data/preprocessed_data/{}'.format(dataset)

    if not os.path.exists(tar_dir):
        os.mkdir(tar_dir)
    if dataset in ['TubeTK', 'BreastMRI', 'IXI']:
        img_modality = 'MRI'
    else:
        img_modality = 'CT'

    preprocessor = Preprocessor(img_dir=img_dir, seg_dir=seg_dir, total_sub_list=total_sub_list,
                                train_sub_list=train_sub_list,
                                tar_dir=tar_dir, img_modality=img_modality)

    preprocessor.run()



