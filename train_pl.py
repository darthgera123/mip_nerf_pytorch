import os, sys
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader

# models
from nerf_mlp import PosEmbedding,NeRF
from rendering import rendering
from dataset import NerfData
from metrics import *
from loss import MSELoss
from tqdm import tqdm 
from torch.optim import AdamW,Adam, lr_scheduler
import cv2
from PIL import Image
import torchvision.transforms as T
import numpy as np

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TestTubeLogger


def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = MSELoss()
        xyz_L = hparams.embed_xyz
        dir_L = hparams.embed_dir
        self.embedding_xyz = PosEmbedding(3, 10) # 10 is the default number
        self.embedding_dir = PosEmbedding(3, 4) # 4 is the default number
        # self.embeddings = [self.embedding_xyz, self.embedding_dir]
        self.embeddings = {}
        self.embeddings['xyz'] = self.embedding_xyz
        self.embeddings['dir'] = self.embedding_dir

        self.nerf_coarse = NeRF(types="coarse",density=8,width=256,skips=[4],in_channels_xyz=3+6*xyz_L,in_channels_dir=3+6*dir_L)
        self.models = {}
        self.models['coarse'] = self.nerf_coarse
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(types="fine",density=8,width=256,skips=[4],in_channels_xyz=3+6*xyz_L,in_channels_dir=3+6*dir_L)
            self.models['fine'] = self.nerf_fine

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                rendering(self.models,self.embeddings,rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,self.hparams.use_disp,
                            self.hparams.perturb,self.hparams.noise_std,
                            self.hparams.N_importance,self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]
        # Only send chunks as input then concatenate all the outputs to get results
        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        self.train_dataset = NerfData(self.hparams.image_dir,self.hparams.img_width,self.hparams.img_height,'train')
        self.val_dataset = NerfData(self.hparams.image_dir,self.hparams.img_width,self.hparams.img_height,'val')

    def configure_optimizers(self):
        self.optimizer = AdamW(list(self.models['coarse'].parameters()) \
            + list(self.models['fine'].parameters()),lr = hparams.lr)
        lr_sched = lr_scheduler.ExponentialLR(self.optimizer,gamma = 1e-5)
        
        return [self.optimizer], [lr_sched]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'],batch['images']
        results = self(rays)
        loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs= batch['rays'], batch['images']
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        loss = self.loss(results, rgbs)
        
        log = {'val_loss': loss}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_width,self.hparams.img_height
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'{hparams.ckpt_dir}/ckpts/{hparams.exp_name}',
                                                                '{epoch:d}'),
                                          monitor='val/psnr',
                                          mode='max',
                                          save_top_k=5,)

    logger = TestTubeLogger(save_dir=os.path.join(hparams.log_dir,hparams.exp_name),
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams.refresh_every,
                      gpus=hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None)


    trainer.fit(system)