from dataclasses import dataclass, field
from typing import Type, Dict, List, Literal
from diconerf.diconerf_loss import SimLoss
from nerfstudio.model_components.renderers import RGBRenderer
from nerfstudio.models.instant_ngp import InstantNGPModelConfig, NGPModel
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

# ViT-L-14 | laion2b_s32b_b82k
# ViT-B-16 | dfn2b
# ViT-L-14-quickgelu |  dfn2b
        
@dataclass
class DiCoModelConfig(InstantNGPModelConfig): 
    use_clip_loss : bool = True
    average_init_density: float = 0.01
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    text_promt: str = "kitti360"
    clip_model: str = "ViT-B-16"
    clip_data : str = "laion2b_s34b_b88k" 
    patch_size : int = 64 

    _target: Type = field(default_factory=lambda: DiCoModel)
    
class DiCoModel(NGPModel):    

    config: DiCoModelConfig

    def populate_modules(self):
        super().populate_modules()

        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.SimLoss = SimLoss(self.config.clip_model, self.config.clip_data)

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        image = batch["image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        if self.training:
            if self.config.use_simloss:
                if self.config.text_promt == "kitti360" :
                    text = ["sky", "car", "black","reflection","road","traffic sign","sidewalk","buildings","vegetation","truck", "house", "fence"]
                if self.config.text_promt == "parkinglot" :
                    text = ["floor", "car","shadow","reflection","door","light","truck", "fireplug", "Entrance", "barricade", "Labacorn", "Pillar","pipe"]                 
                else: # Change the text based on your dataset.
                    text = ["sky", "car"]
                penalty =  self.SimLoss(gt_rgb, pred_rgb, text, self.config.patch_size)
                loss_dict["rgb_loss"] =  penalty * self.rgb_loss(gt_rgb, pred_rgb)
        return loss_dict