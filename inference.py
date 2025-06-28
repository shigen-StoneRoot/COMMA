import inspect
import itertools
import multiprocessing
import os
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from functools import lru_cache
from typing import Union, Tuple, List
from scipy.ndimage import gaussian_filter
from dataloading_new import get_center

def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)

def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@lru_cache(maxsize=2)
def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], sigma_scale: float = 1. / 8,
                     value_scaling_factor: float = 1, dtype=torch.float16, device=torch.device('cuda', 0)) \
        -> torch.Tensor:
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map = gaussian_importance_map / torch.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.type(dtype).to(device)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = torch.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


class nnUNetPredictor(object):
    def __init__(self,
                 patch_size,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.patch_size = patch_size
        self.num_segmentation_heads = 2
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            # device = torch.device(type='cuda', index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!
            # why would I ever want to do that. Stupid dobby. This kills DDP inference...
            pass
        if device.type != 'cuda':
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device
        self.glb_data = None


    def manual_initialization(self, network: nn.Module,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        # self.plans_manager = plans_manager
        # self.configuration_manager = configuration_manager
        # self.list_of_parameters = parameters
        self.network = network
        # self.dataset_json = dataset_json
        # self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        # self.label_manager = plans_manager.get_label_manager(dataset_json)
        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.patch_size) < len(image_size):
            assert len(self.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.patch_size)]]))
        return slicers


    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose: print(f'Input shape: {input_image.shape}')
                if self.verbose: print("step_size:", self.tile_step_size)
                if self.verbose: print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)
                shp = data.shape[1:]
                slicers = self._internal_get_sliding_window_slicers(shp)
                if self.glb_data is not None:
                    self.patch_centers = []
                    for this_slice in slicers:
                        z_start, z_end = this_slice[1].start, this_slice[1].stop
                        y_start, y_end = this_slice[2].start, this_slice[2].stop
                        x_start, x_end = this_slice[3].start, this_slice[3].stop

                        p_center = np.array([(z_start + z_end) / shp[0] - 1,
                                             (y_start + y_end) / shp[1] - 1,
                                             (x_start + x_end) / shp[2] - 1])
                        self.patch_centers.append(p_center)

                if self.perform_everything_on_device and self.device != 'cpu':
                    # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                    try:
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)
                    except RuntimeError:
                        print('Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                        empty_cache(self.device)
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, self.perform_everything_on_device)

                empty_cache(self.device)
                # revert padding
                predicted_logits = predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]
        return predicted_logits


    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        results_device = self.device if do_on_device else torch.device('cpu')
        empty_cache(self.device)

        # move data to device
        if self.verbose:
            print(f'move image to device {results_device}')
        data = data.to(results_device)

        # preallocate arrays
        if self.verbose:
            print(f'preallocating results arrays on device {results_device}')
        predicted_logits = torch.zeros((self.num_segmentation_heads, *data.shape[1:]),
                                       dtype=torch.half,
                                       device=results_device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
        if self.use_gaussian:
            gaussian = compute_gaussian(tuple(self.patch_size), sigma_scale=1. / 8,
                                        value_scaling_factor=10,
                                        device=results_device)

        if self.verbose: print('running prediction')
        if not self.allow_tqdm and self.verbose: print(f'{len(slicers)} steps')
        ith = 0
        for sl in tqdm(slicers, disable=not self.allow_tqdm):
            workon = data[sl][None]
            workon = workon.to(self.device, non_blocking=False)
            if self.glb_data is None:
                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
            else:
                coords = self.patch_centers[ith]
                prediction = self._internal_maybe_mirror_and_predict_with_coord(workon, coords)[0].to(results_device)
                ith += 1

            predicted_logits[sl] += (prediction * gaussian if self.use_gaussian else prediction)
            n_predictions[sl[1:]] += (gaussian if self.use_gaussian else 1)

        predicted_logits /= n_predictions
        # check for infs
        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                               'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                               'predicted_logits to fp32')
        return predicted_logits

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        if self.glb_data is None:
            prediction = self.network(x)
        else:
            prediction = self.network(self.glb_data, x, (0, 0, 0))

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations([m + 2 for m in mirror_axes], i + 1)
            ]
            for axes in axes_combinations:
                if self.glb_data is None:
                    prediction += torch.flip(self.network(torch.flip(x, (*axes,))), (*axes,))
                else:

                    prediction += torch.flip(self.network(torch.flip(self.glb_data, (*axes,)), torch.flip(x, (*axes,))),
                                             (*axes,))
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def _internal_maybe_mirror_and_predict_with_coord(self, x: torch.Tensor, coords) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        if self.glb_data is None:
            prediction = self.network(x)
        else:
            self.network.set_flips_angles([np.array([0, 0, 0])], None)
            prediction = self.network(self.glb_data, x, [coords])

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations([m + 2 for m in mirror_axes], i + 1)
            ]
            flips = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]),
                     np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 1]),
                     np.array([1, 1, 1])]

            for it, axes in enumerate(axes_combinations):
                if self.glb_data is None:
                    prediction += torch.flip(self.network(torch.flip(x, (*axes,))), (*axes,))
                else:
                    self.network.set_flips_angles([flips[it]], None)
                    prediction += torch.flip(self.network(torch.flip(self.glb_data, (*axes,)),
                                                          torch.flip(x, (*axes,)),
                                                          [coords]),
                                             (*axes,))
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def set_glb_data(self, x_glb):
        self.glb_data = x_glb

    def set_glb_coord(self, glb_coord):
        self.network.set_glb_coord(glb_coord)

def init_predictor(model, patch_size, inference_allowed_mirroring_axes, device):
    # inference_allowed_mirroring_axes = (0, 1)
    predictor = nnUNetPredictor(patch_size=patch_size, tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=device, verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(model, inference_allowed_mirroring_axes)

    return predictor

if __name__ == '__main__':
    conv_kernel_sizes = [[3, 3, 3],  [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    pool_op_kernel_sizes = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    n_conv_per_stage_encoder = [2,2,2,2,2,2]
    n_conv_per_stage_decoder = [2,2,2,2,2]

    num_stages = len(conv_kernel_sizes)
    UNet_base_num_features = 32
    unet_max_num_features = 320
    network_class = 'PlainConvUNet'
    conv_or_blocks_per_stage = {
        'n_conv_per_stage': n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': n_conv_per_stage_decoder
    }
    kwargs =  {
            'conv_bias': True,
            'norm_op': nn.InstanceNorm3d,
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }

    from baselines import PlainConvUNet
    model = PlainConvUNet(
        input_channels=1,
        n_stages=num_stages,
        features_per_stage=[min(UNet_base_num_features * 2 ** i,
                                unet_max_num_features) for i in range(num_stages)],
        conv_op=nn.Conv3d,
        kernel_sizes=conv_kernel_sizes,
        strides=pool_op_kernel_sizes,
        num_classes=2,
        deep_supervision=True,
        **conv_or_blocks_per_stage,
        **kwargs
    )
    device = torch.device('cuda:3')
    model = model.to(device)
    model.set_deep_supervision(False)

    inference_allowed_mirroring_axes = mirror_axes = (0, 1, 2)
    predictor = nnUNetPredictor(patch_size=(96, 96, 96), tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=device, verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(model, inference_allowed_mirroring_axes)

    data = torch.rand((1, 128, 448, 448)).to(device)
    logits = predictor.predict_sliding_window_return_logits(data)
    logits = softmax_helper_dim0(logits)
    segmentation = logits.argmax(0)
    print(segmentation.shape)

