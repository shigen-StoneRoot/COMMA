import pickle
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
# from batchgenerators.transforms.spatial_transforms import SpatialTransform
# from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from spatial_transform import SpatialTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from typing import Union, Tuple, List
import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import SimpleITK as sitk
from batchgenerators.dataloading.data_loader import DataLoader
import os
from skimage.transform import resize

def augment_mirroring(sample_data, sample_seg=None, glb_data=None, axes=(0, 1, 2)):
    if (len(sample_data.shape) != 3) and (len(sample_data.shape) != 4):
        raise Exception(
            "Invalid dimension for sample_data and sample_seg. sample_data and sample_seg should be either "
            "[channels, x, y] or [channels, x, y, z]")
    flips = np.array([0, 0, 0])
    if 0 in axes and np.random.uniform() < 0.5:
        sample_data[:, :] = sample_data[:, ::-1]
        if sample_seg is not None:
            sample_seg[:, :] = sample_seg[:, ::-1]
        if glb_data is not None:
            glb_data[:, :] = glb_data[:, ::-1]
        flips[0] = 1
    if 1 in axes and np.random.uniform() < 0.5:
        sample_data[:, :, :] = sample_data[:, :, ::-1]
        if sample_seg is not None:
            sample_seg[:, :, :] = sample_seg[:, :, ::-1]
        if glb_data is not None:
            glb_data[:, :, :] = glb_data[:, :, ::-1]
        flips[1] = 1
    if 2 in axes and len(sample_data.shape) == 4:
        if np.random.uniform() < 0.5:
            sample_data[:, :, :, :] = sample_data[:, :, :, ::-1]
            if sample_seg is not None:
                sample_seg[:, :, :, :] = sample_seg[:, :, :, ::-1]
            if glb_data is not None:
                glb_data[:, :, :, :] = glb_data[:, :, :, ::-1]
            flips[2] = 1
    return sample_data, sample_seg, glb_data, flips


class MirrorTransform(AbstractTransform):
    """ Randomly mirrors data along specified axes. Mirroring is evenly distributed. Probability of mirroring along
    each axis is 0.5

    Args:
        axes (tuple of int): axes along which to mirror

    """

    def __init__(self, axes=(0, 1, 2), data_key="data", label_key="seg", p_per_sample=1, glb_data_key='glb_data'):
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.glb_data_key = glb_data_key
        self.label_key = label_key
        self.axes = axes
        if max(axes) > 2:
            raise ValueError("MirrorTransform now takes the axes as the spatial dimensions. What previously was "
                             "axes=(2, 3, 4) to mirror along all spatial dimensions of a 5d tensor (b, c, x, y, z) "
                             "is now axes=(0, 1, 2). Please adapt your scripts accordingly.")

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)
        glb_data = data_dict.get(self.glb_data_key)
        flips_all = data_dict.get('flips')
        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                sample_seg = None
                if seg is not None:
                    sample_seg = seg[b]
                ret_val = augment_mirroring(data[b], sample_seg, glb_data[b], axes=self.axes)
                data[b] = ret_val[0]
                if seg is not None:
                    seg[b] = ret_val[1]
                if glb_data is not None:
                    glb_data[b] = ret_val[2]
                flips_all[b] = ret_val[3]


        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg
        if glb_data is not None:
            data_dict[self.glb_data_key] = glb_data

        data_dict['flips'] = flips_all
        return data_dict


def compute_bounding_box(array):
    # 获取非零元素的索引
    non_zero_indices = np.transpose(np.nonzero(array))

    # 计算包含所有非零元素的最小方框
    min_coords = np.min(non_zero_indices, axis=0)
    max_coords = np.max(non_zero_indices, axis=0)

    return min_coords, max_coords


def normlize(image):
    mean = image.mean()
    std = image.std()
    image = (image - mean) / (max(std, 1e-8))
    return image


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)


class nnUNetDataset(object):
    def __init__(self, img_list, gt_list, mode='train'):

        super().__init__()
        self.mode = mode
        assert self.mode in ['train', 'test']
        self.dataset = {}
        self.img_list, self.gt_list = img_list, gt_list
        self.subs = [gt.split('/')[-1] for gt in gt_list]
        self.bounding_box = {}
        self.ori_shape = {}
        self.class_locs = {}
        for i in range(len(self.gt_list)):
            self.bounding_box[i] = []
            self.ori_shape[i] = []
            self.class_locs[i] = None

    def __getitem__(self, idx):
        img_path, gt_path = self.img_list[idx], self.gt_list[idx]
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(float)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path)).astype(int)

        if len(self.bounding_box[idx]) == 0:
            min_coords, max_coords = compute_bounding_box(img)
            self.bounding_box[idx].append(min_coords)
            self.bounding_box[idx].append(max_coords)
        else:
            min_coords, max_coords = self.bounding_box[idx]

        if len(self.ori_shape[idx]) == 0:
            self.ori_shape[idx].extend([img.shape[0], img.shape[1], img.shape[2]])

        img = img[min_coords[0]:max_coords[0] + 1, min_coords[1]:max_coords[1] + 1,
                  min_coords[2]:max_coords[2] + 1]
        gt = gt[min_coords[0]:max_coords[0] + 1, min_coords[1]:max_coords[1] + 1,
                min_coords[2]:max_coords[2] + 1]

        if self.class_locs[idx] is None:
            self.class_locs[idx] = sample_foreground_locations(gt[np.newaxis, :], [1])
        img = normlize(img)
        return img, gt

    def load_case(self, idx):
        data, seg = self.__getitem__(idx)
        if self.mode == 'train':
            return data[np.newaxis, :], seg[np.newaxis, :], None
        else:
            return torch.from_numpy(data).float().unsqueeze(0), \
                   torch.from_numpy(seg).long().unsqueeze(0), \
                   self.bounding_box[idx], self.ori_shape[idx]

    def get_test_idx(self):
        assert self.mode == 'test'
        return np.arange(len(self.gt_list))


# class nnUNetDataLoaderBase(DataLoader):
#     def __init__(self,
#                  data: nnUNetDataset,
#                  batch_size: int,
#                  patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
#                  final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
#                  oversample_foreground_percent: float = 0.0,
#                  sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
#                  pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
#                  probabilistic_oversampling: bool = False):
#         super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)
#         assert isinstance(data, nnUNetDataset), 'nnUNetDataLoaderBase only supports dictionaries as data'
#
#         self.indices = np.arange(len(data.subs))
#         self.oversample_foreground_percent = oversample_foreground_percent
#         self.final_patch_size = final_patch_size
#         self.patch_size = patch_size
#         # self.list_of_keys = list(self._data.keys())
#         # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
#         # (which is what the network will get) these patches will also cover the border of the images
#         self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
#         if pad_sides is not None:
#             if not isinstance(pad_sides, np.ndarray):
#                 pad_sides = np.array(pad_sides)
#             self.need_to_pad += pad_sides
#         self.num_channels = None
#         self.pad_sides = pad_sides
#         self.data_shape, self.seg_shape = self.determine_shapes()
#         self.sampling_probabilities = sampling_probabilities
#         self.annotated_classes_key = (0, 1)
#         self.has_ignore = False
#         self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
#             else self._probabilistic_oversampling
#
#     def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
#         """
#         determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
#         """
#         return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))
#
#     def _probabilistic_oversampling(self, sample_idx: int) -> bool:
#         # print('YEAH BOIIIIII')
#         return np.random.uniform() < self.oversample_foreground_percent
#
#     def determine_shapes(self):
#         # load one case
#         data, seg, properties = self._data.load_case(self.indices[0])
#         num_color_channels = data.shape[0]
#
#         data_shape = (self.batch_size, num_color_channels, *self.patch_size)
#         seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
#         return data_shape, seg_shape
#
#     def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
#                  overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
#         # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
#         # locations for the given slice
#         need_to_pad = self.need_to_pad.copy()
#         dim = len(data_shape)
#
#         for d in range(dim):
#             # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
#             # always
#             if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
#                 need_to_pad[d] = self.patch_size[d] - data_shape[d]
#
#         # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
#         # define what the upper and lower bound can be to then sample form them with np.random.randint
#         lbs = [- need_to_pad[i] // 2 for i in range(dim)]
#         ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]
#
#         # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
#         # at least one of the foreground classes in the patch
#         if not force_fg and not self.has_ignore:
#             bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
#             # print('I want a random location')
#         else:
#             if not force_fg and self.has_ignore:
#                 selected_class = self.annotated_classes_key
#                 if len(class_locations[selected_class]) == 0:
#                     # no annotated pixels in this case. Not good. But we can hardly skip it here
#                     print('Warning! No annotated pixels in image!')
#                     selected_class = None
#                 # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
#             elif force_fg:
#                 assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
#                 if overwrite_class is not None:
#                     assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
#                                                                       'have class_locations (missing key)'
#                 # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
#                 # class_locations keys can also be tuple
#                 eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
#
#                 # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
#                 # strange formulation needed to circumvent
#                 # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
#                 tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
#                 if any(tmp):
#                     if len(eligible_classes_or_regions) > 1:
#                         eligible_classes_or_regions.pop(np.where(tmp)[0][0])
#
#                 if len(eligible_classes_or_regions) == 0:
#                     # this only happens if some image does not contain foreground voxels at all
#                     selected_class = None
#                     if verbose:
#                         print('case does not contain any foreground classes')
#                 else:
#                     # I hate myself. Future me aint gonna be happy to read this
#                     # 2022_11_25: had to read it today. Wasn't too bad
#                     selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
#                         (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
#                 # print(f'I want to have foreground, selected class: {selected_class}')
#             else:
#                 raise RuntimeError('lol what!?')
#             voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None
#
#             if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
#                 selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
#                 # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
#                 # Make sure it is within the bounds of lb and ub
#                 # i + 1 because we have first dimension 0!
#                 bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
#             else:
#                 # If the image does not contain any foreground classes, we fall back to random cropping
#                 bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
#
#         bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
#
#         return bbox_lbs, bbox_ubs

class nnUNetDataLoaderBase(DataLoader):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 glb_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False):
        super().__init__(data, batch_size, 1, None, True, False, True, sampling_probabilities)
        assert isinstance(data, nnUNetDataset), 'nnUNetDataLoaderBase only supports dictionaries as data'

        self.indices = np.arange(len(data.subs))
        self.oversample_foreground_percent = oversample_foreground_percent
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size
        self.glb_size = glb_size
        # self.list_of_keys = list(self._data.keys())
        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the images
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()
        self.sampling_probabilities = sampling_probabilities
        self.annotated_classes_key = (0, 1)
        self.has_ignore = False
        self.get_do_oversample = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling

    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """
        determines whether sample sample_idx in a minibatch needs to be guaranteed foreground
        """
        return not sample_idx < round(self.batch_size * (1 - self.oversample_foreground_percent))

    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        # print('YEAH BOIIIIII')
        return np.random.uniform() < self.oversample_foreground_percent

    def determine_shapes(self):
        # load one case
        data, seg, properties = self._data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape

    def get_bbox(self, data_shape: np.ndarray, force_fg: bool, class_locations: Union[dict, None],
                 overwrite_class: Union[int, Tuple[int, ...]] = None, verbose: bool = False):
        # in dataloader 2d we need to select the slice prior to this and also modify the class_locations to only have
        # locations for the given slice
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i] for i in range(dim)]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
            # print('I want a random location')
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    # no annotated pixels in this case. Not good. But we can hardly skip it here
                    print('Warning! No annotated pixels in image!')
                    selected_class = None
                # print(f'I have ignore labels and want to pick a labeled area. annotated_classes_key: {self.annotated_classes_key}')
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                # class_locations keys can also be tuple
                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]

                # if we have annotated_classes_key locations and other classes are present, remove the annotated_classes_key from the list
                # strange formulation needed to circumvent
                # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp):
                    if len(eligible_classes_or_regions) > 1:
                        eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    # this only happens if some image does not contain foreground voxels at all
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    # I hate myself. Future me aint gonna be happy to read this
                    # 2022_11_25: had to read it today. Wasn't too bad
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or (overwrite_class not in eligible_classes_or_regions)) else overwrite_class
                # print(f'I want to have foreground, selected class: {selected_class}')
            else:
                raise RuntimeError('lol what!?')
            voxels_of_that_class = class_locations[selected_class] if selected_class is not None else None

            if voxels_of_that_class is not None and len(voxels_of_that_class) > 0:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                # i + 1 because we have first dimension 0!
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.patch_size[i] // 2) for i in range(dim)]
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]

        return bbox_lbs, bbox_ubs


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch_with_idx(self, selected_keys):
        # selected_keys = self.get_indices()

        # print(self.indices)
        # selected_keys = np.random.choice(self.indices, self.batch_size, replace=False)
        # print(selected_keys)

        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        glb_data_all = np.zeros((self.data_shape[0], self.data_shape[1],
                                 self.glb_size[0], self.glb_size[1], self.glb_size[2]), dtype=np.float32)
        case_properties = []
        coors_all = []
        flips_all = []
        angles_all = []

        for j, i in enumerate(selected_keys):

            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            for c in range(glb_data_all.shape[1]):
                glb_data_all[j, c] = resize(data[c], self.glb_size, order=1)
            class_locations = self._data.class_locs[i]

            case_properties.append(properties)


            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, class_locations)


            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]


            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            z_start, z_end = this_slice[1].start, this_slice[1].stop
            y_start, y_end = this_slice[2].start, this_slice[2].stop
            x_start, x_end = this_slice[3].start, this_slice[3].stop

            p1 = np.array([0,
                           2 * z_start / shape[0] - 1,
                           2 * y_start / shape[1] - 1,
                           2 * x_end / shape[2] - 1])
            p2 = np.array([0,
                           2 * z_end / shape[0] - 1,
                           2 * y_start / shape[1] - 1,
                           2 * x_start / shape[2] - 1])
            p3 = np.array([0,
                           2 * z_end / shape[0] - 1,
                           2 * y_end / shape[1] - 1,
                           2 * x_end / shape[2] - 1])
            p_center = np.array([0,
                                 (z_start + z_end) / shape[0] - 1,
                                 (y_start + y_end) / shape[1] - 1,
                                 (x_start + x_end) / shape[2] - 1])

            coor = np.concatenate([p1[np.newaxis, :],
                                   p2[np.newaxis, :],
                                   p3[np.newaxis, :],
                                   p_center[np.newaxis, :]], 0)
            coors_all.append(coor)
            flips_all.append(np.array([0, 0, 0]))
            angles_all.append(np.array([0, 0, 0]))

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return {'data': data_all, 'seg': seg_all, 'glb_data': glb_data_all,
                'coors': coors_all, 'flips': flips_all, 'angles': angles_all,
                'properties': case_properties, 'keys': selected_keys}

    # def generate_train_batch(self):
    #     selected_keys = self.get_indices()
    #
    #     # print(self.indices)
    #     # selected_keys = np.random.choice(self.indices, self.batch_size, replace=False)
    #     print(selected_keys)
    #
    #     # preallocate memory for data and seg
    #     data_all = np.zeros(self.data_shape, dtype=np.float32)
    #     seg_all = np.zeros(self.seg_shape, dtype=np.int16)
    #     case_properties = []
    #
    #     for j, i in enumerate(selected_keys):
    #
    #         force_fg = self.get_do_oversample(j)
    #
    #         data, seg, properties = self._data.load_case(i)
    #
    #         all_class_locations = np.argwhere(seg == 1)
    #         np.random.seed(i)
    #         selected_class_locations = all_class_locations[np.random.choice(all_class_locations.shape[0],
    #                                                        10000, replace=False)]
    #         class_locations = {}
    #         class_locations[1] = selected_class_locations
    #         case_properties.append(properties)
    #
    #
    #         shape = data.shape[1:]
    #         dim = len(shape)
    #         bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, class_locations)
    #
    #
    #         valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
    #         valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
    #
    #
    #         this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
    #         data = data[this_slice]
    #
    #         this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
    #         seg = seg[this_slice]
    #
    #         padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
    #         data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
    #         seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)
    #
    #     return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


class AugmentDataloader(object):
    def __init__(self, data_loader, transform):
        self.data_loader = data_loader
        self.transform = transform
        self.indices = self.data_loader.indices.copy()
        np.random.shuffle(self.indices)
        self.it = 0
        self.batch_size = self.data_loader.batch_size
        self.one_epoch_its = len(self.indices) // self.batch_size

    def __iter__(self):
        return self

    def __next__(self):
        # if self.it == self.one_epoch_its + 1:
        #     self.it = 0
        #     np.random.shuffle(self.indices)
        # selected_keys = self.indices[self.it * self.batch_size:
        #                              min((self.it + 1) * self.batch_size, len(self.indices))]
        # if len(selected_keys) == 1:
        #     selected_keys = np.concatenate([selected_keys, self.indices[:self.batch_size - 1]], 0)
        #     assert len(selected_keys) == self.batch_size
        # self.it += 1
        np.random.shuffle(self.indices)
        selected_keys = np.random.choice(self.indices, self.batch_size)
        item = self.data_loader.generate_train_batch_with_idx(selected_keys)
        # item = next(self.data_loader)
        if self.transform is not None:
            item = self.transform(**item)
        return item

    def next(self):
        return self.__next__()

class DownsampleSegForDSTransform2(AbstractTransform):
    '''
    data_dict['output_key'] will be a list of segmentations scaled according to ds_scales
    '''
    def __init__(self, ds_scales: Union[List, Tuple],
                 order: int = 0, input_key: str = "seg",
                 output_key: str = "seg", axes: Tuple[int] = None):
        """
        Downscales data_dict[input_key] according to ds_scales. Each entry in ds_scales specified one deep supervision
        output and its resolution relative to the original data, for example 0.25 specifies 1/4 of the original shape.
        ds_scales can also be a tuple of tuples, for example ((1, 1, 1), (0.5, 0.5, 0.5)) to specify the downsampling
        for each axis independently
        """
        self.axes = axes
        self.output_key = output_key
        self.input_key = input_key
        self.order = order
        self.ds_scales = ds_scales

    def __call__(self, **data_dict):
        if self.axes is None:
            axes = list(range(2, data_dict[self.input_key].ndim))
        else:
            axes = self.axes

        output = []
        for s in self.ds_scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * len(axes)
            else:
                assert len(s) == len(axes), f'If ds_scales is a tuple for each resolution (one downsampling factor ' \
                                            f'for each axis) then the number of entried in that tuple (here ' \
                                            f'{len(s)}) must be the same as the number of axes (here {len(axes)}).'

            if all([i == 1 for i in s]):
                output.append(data_dict[self.input_key])
            else:
                new_shape = np.array(data_dict[self.input_key].shape).astype(float)
                for i, a in enumerate(axes):
                    new_shape[a] *= s[i]
                new_shape = np.round(new_shape).astype(int)
                out_seg = np.zeros(new_shape, dtype=data_dict[self.input_key].dtype)
                for b in range(data_dict[self.input_key].shape[0]):
                    for c in range(data_dict[self.input_key].shape[1]):
                        out_seg[b, c] = resize_segmentation(data_dict[self.input_key][b, c], new_shape[2:], self.order)
                output.append(out_seg)
        data_dict[self.output_key] = output
        return data_dict


# def get_training_transforms(
#         patch_size: Union[np.ndarray, Tuple[int]],
#         rotation_for_DA: dict,
#         deep_supervision_scales: Union[List, Tuple, None],
#         mirror_axes: Tuple[int, ...],
#         order_resampling_data: int = 3,
#         order_resampling_seg: int = 1,
#         border_val_seg: int = -1,
# ) -> AbstractTransform:
#     tr_transforms = []
#
#     patch_size_spatial = patch_size
#     ignore_axes = None
#
#     tr_transforms.append(SpatialTransform(
#         patch_size_spatial, patch_center_dist_from_border=None,
#         do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
#         do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
#         p_rot_per_axis=1,  # todo experiment with this
#         do_scale=True, scale=(0.7, 1.4),
#         border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
#         border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
#         random_crop=False,  # random cropping is part of our dataloaders
#         p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
#         independent_scale_for_each_axis=False  # todo experiment with this
#     ))
#
#     tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
#     tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
#                                                p_per_channel=0.5))
#     tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
#     tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
#     tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
#                                                         p_per_channel=0.5,
#                                                         order_downsample=0, order_upsample=3, p_per_sample=0.25,
#                                                         ignore_axes=ignore_axes))
#     tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
#     tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))
#
#     if mirror_axes is not None and len(mirror_axes) > 0:
#         tr_transforms.append(MirrorTransform(mirror_axes))
#
#     tr_transforms.append(RemoveLabelTransform(-1, 0))
#     tr_transforms.append(RenameTransform('seg', 'target', True))
#     if deep_supervision_scales is not None:
#         tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
#                                                           output_key='target'))
#     tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
#     tr_transforms = Compose(tr_transforms)
#     return tr_transforms

def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
) -> AbstractTransform:
    tr_transforms = []

    patch_size_spatial = patch_size
    ignore_axes = None

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
        p_rot_per_axis=1,  # todo experiment with this
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
        random_crop=False,  # random cropping is part of our dataloaders
        p_el_per_sample=0, p_scale_per_sample=0, p_rot_per_sample=0,
        # p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False  # todo experiment with this
    ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    # if mirror_axes is not None and len(mirror_axes) > 0:
    #     tr_transforms.append(MirrorTransform(mirror_axes))

    # begin glb transform
    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key='glb_data'))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5, data_key='glb_data'))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15, data_key='glb_data'))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key='glb_data'))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes, data_key='glb_data'))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1, data_key='glb_data'))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3, data_key='glb_data'))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes, glb_data_key='glb_data'))

    # end glb transform



    tr_transforms.append(RemoveLabelTransform(-1, 0))

    tr_transforms.append(RenameTransform('seg', 'target', True))
    if deep_supervision_scales is not None:
        tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                          output_key='target'))
    tr_transforms.append(NumpyToTensor(['data', 'target', 'glb_data'], 'float'))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
    ) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))
        val_transforms.append(RenameTransform('seg', 'target', True))

        if deep_supervision_scales is not None:
            val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                               output_key='target'))
        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms


# def init_dataloader(img_list, gt_list, transform, patch_size=(96, 96, 96)):
#     rotation_for_DA = {
#         'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
#         'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
#         'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
#     }
#
#     initial_patch_size = get_patch_size(patch_size,
#                                         *rotation_for_DA.values(),
#                                         (0.85, 1.25))
#     ds = nnUNetDataset(img_list, gt_list)
#     loader = nnUNetDataLoader3D(ds, batch_size=2,
#                                 patch_size=initial_patch_size,
#                                 final_patch_size=patch_size,
#                                 oversample_foreground_percent=0.33,
#                                 sampling_probabilities=None, pad_sides=None)
#     dataloader = AugmentDataloader(loader, transform)
#
#     return dataloader

def init_training_dataloader(img_list, gt_list, patch_size=(96, 96, 96), glb_size=(96, 160, 160)):
    rotation_for_DA = {
        'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    }
    # mirror_axes = (0, 1)
    mirror_axes = (0, 1, 2)

    deep_supervision_scales = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                               [0.125, 0.125, 0.125], [0.0625, 0.0625, 0.0625]]

    initial_patch_size = get_patch_size(patch_size,
                                        *rotation_for_DA.values(),
                                        (0.85, 1.25))

    ds = nnUNetDataset(img_list, gt_list)
    loader = nnUNetDataLoader3D(ds, batch_size=2,
                                patch_size=initial_patch_size,
                                final_patch_size=patch_size,
                                oversample_foreground_percent=0.33,
                                sampling_probabilities=None, pad_sides=None, glb_size=glb_size)
    training_transforms = get_training_transforms(
        patch_size=patch_size,
        rotation_for_DA=rotation_for_DA,
        deep_supervision_scales=deep_supervision_scales,
        mirror_axes=mirror_axes,
        order_resampling_data=3, order_resampling_seg=1,
        border_val_seg=-1)
    dataloader = AugmentDataloader(loader, training_transforms)

    return dataloader


def init_validation_dataloader(img_list, gt_list):
    dataloader = nnUNetDataset(img_list, gt_list, mode='test')
    return dataloader

if __name__ == '__main__':
    patch_size = (96, 96, 96)
    rotation_for_DA = {
        'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
    }
    img_dir = r'/home/genshi/nnUNet/nnUNet_raw_data_base/Dataset301_TubeTK/imagesTr'
    gt_dir = r'/home/genshi/nnUNet/nnUNet_raw_data_base/Dataset301_TubeTK/labelsTr'

    train_subs = ["Normal002-MRA", "Normal003-MRA", "Normal006-MRA", "Normal008-MRA", "Normal009-MRA", "Normal010-MRA",
                  "Normal011-MRA", "Normal017-MRA", "Normal018-MRA", "Normal020-MRA", "Normal021-MRA", "Normal022-MRA",
                  "Normal023-MRA", "Normal026-MRA", "Normal027-MRA", "Normal033-MRA", "Normal037-MRA", "Normal042-MRA",
                  "Normal043-MRA", "Normal045-MRA", "Normal047-MRA", "Normal054-MRA", "Normal056-MRA", "Normal057-MRA",
                  "Normal058-MRA", "Normal064-MRA", "Normal070-MRA", "Normal074-MRA", "Normal077-MRA", "Normal079-MRA",
                  "Normal082-MRA", "Normal086-MRA", "Normal088-MRA"]

    train_img = [os.path.join(img_dir, sub + '_0000.nii.gz') for sub in train_subs]
    train_gt = [os.path.join(gt_dir, sub + '.nii.gz') for sub in train_subs]

    initial_patch_size = get_patch_size(patch_size,
                                        *rotation_for_DA.values(),
                                        (0.85, 1.25))


    ds = nnUNetDataset(train_img, train_gt)  # this should not load the properties!

    dl_tr = nnUNetDataLoader3D(ds, batch_size=2,
                               patch_size=initial_patch_size,
                               final_patch_size=patch_size,
                               glb_size=[128, 448, 448],
                               oversample_foreground_percent=0.33,
                               sampling_probabilities=None, pad_sides=None)

    item = dl_tr.generate_train_batch_with_idx([1, 1])
    print(item.keys())
    print(item['data'].shape, item['seg'].shape, item['glb_data'].shape)
    coors_all = item['coors']
    glb_size = item['glb_data'].shape[2:]

    # pool_op_kernel_sizes = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]

    deep_supervision_scales = [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25],
                               [0.125, 0.125, 0.125], [0.0625, 0.0625, 0.0625]]

    mirror_axes = (0, 1, 2)

    training_transforms = get_training_transforms(
        patch_size=patch_size,
        rotation_for_DA=rotation_for_DA,
        deep_supervision_scales=deep_supervision_scales,
        mirror_axes=mirror_axes,
        order_resampling_data=3, order_resampling_seg=1,
        border_val_seg=-1)


    aug_data = training_transforms(**item)


    def convert_relative_to_absolute(points, shape):
        center = np.array(shape)
        absolute_points = []
        for point in points:
            absolute_point = np.round((np.array(point) + 1) / 2 * center)
            absolute_points.append(absolute_point)
        return absolute_points


    def flip_points(points, flips, shape):
        flipped_points = []
        for point in points:
            z, y, x = point
            if flips[0] == 1:
                z = shape[0] - 1 - point[0]
            if flips[1] == 1:
                y = shape[1] - 1 - point[1]
            if flips[2] == 1:
                x = shape[2] - 1 - point[2]
            flipped_points.append((z, y, x))
        return flipped_points


    def rotate_points(points, center, angles):
        a, b, c = angles
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
        R = Rx @ Ry @ Rz

        # 平移到原点进行旋转，然后再平移回原位置
        rotated_points = [R @ (point - center) + center for point in points]
        return np.array(rotated_points)


    def crop_data(X, relative_points, flips, angles):
        # 将相对坐标转换为绝对坐标
        absolute_points = convert_relative_to_absolute(relative_points, X.shape)

        # 旋转坐标
        center = np.array(X.shape) / 2
        rotated_points = rotate_points(absolute_points, center, angles)

        # 翻转坐标
        flipped_points = flip_points(rotated_points, flips, X.shape)



        # 计算剪裁区域
        zs, ys, xs = zip(*flipped_points)
        z_min, z_max = max(0, min(zs)), min(X.shape[0] - 1, max(zs))
        y_min, y_max = max(0, min(ys)), min(X.shape[1] - 1, max(ys))
        x_min, x_max = max(0, min(xs)), min(X.shape[2] - 1, max(xs))

        center = [(z_min + z_max) / 2, (y_min + y_max) / 2, (x_min + x_max) / 2]

        rate = 0.376623
        z_half = round((z_max - z_min + 1) * rate / 2)
        y_half = round((y_max - y_min + 1) * rate / 2)
        x_half = round((x_max - x_min + 1) * rate / 2)

        # 剪裁数据
        cropped_data = X[int(z_min):int(z_max) + 1, int(y_min):int(y_max) + 1, int(x_min):int(x_max) + 1]

        cropped_data = cropped_data[z_half:-z_half, y_half:-y_half, x_half:-x_half]
        return cropped_data, center


    # # 示例
    # X = np.random.rand(10, 10, 10)  # 3D 数据
    # flips = np.array([1, 0, 1])  # 翻转数组
    # angles = (np.radians(30), np.radians(45), np.radians(60))  # 旋转角度（弧度制）
    # original_points = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]  # 原始坐标点
    #
    # cropped_data = crop_data(X, original_points, flips, angles)


    # def crop_flipped_data(X, relative_points, flips):
    #     # 获取数据的维度
    #     Z_max, Y_max, X_max = X.shape
    #
    #     # 计算中心位置
    #     center = np.array([Z_max / 2, Y_max / 2, X_max / 2])
    #
    #     # 转换相对坐标到实际索引位置
    #     mapped_points = []
    #     for _, z, y, x in relative_points:
    #         # 映射到实际索引
    #         real_z = int((z + 1) / 2 * (Z_max - 1))
    #         real_y = int((y + 1) / 2 * (Y_max - 1))
    #         real_x = int((x + 1) / 2 * (X_max - 1))
    #
    #         # 应用翻转
    #         if flips[0]:  # Z轴翻转
    #             real_z = Z_max - 1 - real_z
    #         if flips[1]:  # Y轴翻转
    #             real_y = Y_max - 1 - real_y
    #         if flips[2]:  # X轴翻转
    #             real_x = X_max - 1 - real_x
    #
    #         mapped_points.append((real_z, real_y, real_x))
    #
    #     # 确定剪裁区域
    #     zs, ys, xs = zip(*mapped_points)
    #     z_min, z_max = min(zs), max(zs)
    #     y_min, y_max = min(ys), max(ys)
    #     x_min, x_max = min(xs), max(xs)
    #
    #     # 剪裁数据
    #     cropped_data = X[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
    #     return cropped_data

    print(aug_data['data'].shape, aug_data['glb_data'].shape)

    p1, p2, p3, p_center = coors_all[0]

    flips = aug_data['flips'][0]
    angles = aug_data['angles'][0]
    print(flips)
    print(angles)

    # print(p_center[1:])


    # 示例数据
    X = aug_data['glb_data'][0].numpy().squeeze()  # 代表翻转后的数据
    relative_points = [p1[1:], p2[1:], p3[1:]]  # 相对坐标点

    p_center = convert_relative_to_absolute([p_center[1:]], X.shape)[0]
    rotator_center = rotate_points([p_center], np.array(X.shape) / 2, angles)[0]
    flip_center = flip_points([rotator_center], flips, X.shape)[0]


    # 获取剪裁后的数据
    cropped_data, center2 = crop_data(X, relative_points, flips, angles)
    print(cropped_data.shape)
    # print(center2)
    import matplotlib.pyplot as plt

    plt.imshow(aug_data['data'][0].numpy().squeeze().max(0), cmap='gray')
    plt.show()

    plt.imshow(cropped_data.squeeze().max(0), cmap='gray')
    plt.show()

    plt.imshow(aug_data['glb_data'][0].numpy().squeeze().max(0), cmap='gray')
    plt.scatter([flip_center[2]], [flip_center[1]], c='r', s=20)
    # plt.scatter([center2[2]], [center2[1]], c='green', s=20)
    plt.show()

    plt.imshow(aug_data['data'][0].numpy().squeeze().max(1), cmap='gray')
    plt.show()

    plt.imshow(cropped_data.squeeze().max(1), cmap='gray')
    plt.show()

    plt.imshow(aug_data['glb_data'][0].numpy().squeeze().max(1), cmap='gray')
    plt.show()

    plt.imshow(aug_data['data'][0].numpy().squeeze().max(2), cmap='gray')
    plt.show()

    plt.imshow(cropped_data.squeeze().max(2), cmap='gray')
    plt.show()

    plt.imshow(aug_data['glb_data'][0].numpy().squeeze().max(2), cmap='gray')
    plt.show()




    for t in aug_data['target']:
        print(t.shape)