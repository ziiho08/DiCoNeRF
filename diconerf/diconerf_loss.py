import open_clip
from PIL import Image
import torch
from torch import nn
from typing import Literal
import torch.nn.functional as F

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss
cross_entropy = F.cross_entropy
kl_divergence = nn.KLDivLoss(reduction="batchmean")

class SimLoss(nn.Module):
    
    def reshape_(self, output,patch_size):
        if output.shape[0] > patch_size*patch_size:
            output = output[:patch_size*patch_size]
        output = output.view(patch_size, patch_size, 3)
        output = output.permute(2, 0, 1)
        output = Image.fromarray(output.detach().cpu().numpy().astype('uint8').transpose(1, 2, 0))
        return output       
    
    def __init__(self, clip_model, clip_data, reduction_type: Literal["image", "batch"] = "batch"):
        super().__init__()
        self.reduction_type: Literal["image", "batch"] = reduction_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_data = clip_data
        self.clip_model = clip_model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(self.clip_model, pretrained=self.clip_data, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(self.clip_model)
        
    def forward(self, gt_rgb, pred_rgb, text_list, patch_size):
                
        pred_image = self.preprocess(self.reshape_(pred_rgb,patch_size)).unsqueeze(0).to(self.device)
        gt_image = self.preprocess(self.reshape_(gt_rgb,patch_size)).unsqueeze(0).to(self.device)
        text = self.tokenizer(["A cropped photo of the " + desc for desc in text_list]).to(self.device)

        with torch.autocast(device_type=self.device):
            with torch.no_grad():
              pred_embeddings = self.model.encode_image(pred_image) #[11,512]
              gt_embeddings = self.model.encode_image(gt_image) 
              text_embeddings = self.model.encode_text(text) 
              
              pred_embeddings_ = pred_embeddings / pred_embeddings.norm(dim=-1, keepdim=True)
              gt_embeddings_ = gt_embeddings / gt_embeddings.norm(dim=-1, keepdim=True)
              text_embeddings_ = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
  
              pred_similarity = (pred_embeddings_ @ text_embeddings_.T)
              gt_similarity = (gt_embeddings_ @ text_embeddings_.T)
  
              lambda_ = 0.1
              kl = kl_divergence(F.log_softmax((100* pred_similarity), dim=-1), F.softmax(gt_similarity, dim=-1))
              penalty = (2 - torch.exp(-lambda_ * kl))
      
              return penalty
        
