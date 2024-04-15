from __future__ import annotations

from diconerf.diconerf_datamanager import (
    DiCoDataManagerConfig)
from diconerf.diconerf_model import DiCoModelConfig
from diconerf.diconerf_pipeline import (
    DiCoPipelineConfig)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig


diconerf = MethodSpecification(
    config=TrainerConfig(
        method_name="diconerf",  
        steps_per_eval_batch=500,
        steps_per_save=17500,
        max_num_iterations=35001,
        steps_per_eval_all_images = 17500,
        mixed_precision=True,
        gradient_accumulation_steps = {'proposal_networks':512, 'camera_opt':512},
        save_only_latest_checkpoint=False,
        pipeline=DiCoPipelineConfig(
            datamanager=DiCoDataManagerConfig(
                pixel_sampler=PatchPixelSamplerConfig(
                    patch_size=64,
                    num_rays_per_batch=4096,
                    ignore_mask = False, 
                    rejection_sample_mask = True,
                    fisheye_crop_radius = None
                ),
                dataparser=NerfstudioDataParserConfig(
                    downscale_factor=4, 
                    train_split_fraction=0.8,
                    load_3D_points=True,
                    scene_scale = 1,
                    scale_factor = 1,
                    orientation_method = "none",
                    center_method = "none",
                    depth_unit_scale_factor=0.001,
                ),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                masks_on_gpu=True,
                images_on_gpu=True,
                patch_size=64,
                ),
            model=DiCoModelConfig( 
                eval_num_rays_per_chunk=1 << 12,
                num_nerf_samples_per_ray=128,
                use_gradient_scaling=False,
                near_plane = 0.05,
                far_plane = 1000.0
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15), 
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 12),
        vis="wandb",
    ),
    description="DiCo-NeRF method.",
)