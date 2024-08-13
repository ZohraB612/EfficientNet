import wandb
from Data_Set import my_collate, Tensor
from models import SetTransformer, CachedMLP, srm
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from diffusers import DDIMScheduler, DDPMScheduler
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from utils import draw, sample

class EnhancedEfficientSRM(srm):
    def __init__(self, *args, steps_per_epoch=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_per_epoch = steps_per_epoch  # Ensure this is set correctly
        self.automatic_optimization = False
        self.grad_clip_val = 0.5

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()

        with torch.cuda.amp.autocast():
            loss = self.compute_loss(batch)

        self.manual_backward(loss)

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)
        opt.step()
        sch.step()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be provided and not None.")
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.steps_per_epoch * 10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.compute_loss(batch)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            Set = batch[0]
            condition, mu, sigma = self.encoder(Set)
            for i in range(mu.shape[0]):
                filename = f'Results/{self.experiment_name}/{self.current_epoch}_{batch_idx}_{i}.svg'
                stroke = sample(self.samples, self.sample_steps, self.decoder, self.noise_scheduler_sample, mu[i], self.dim_in)
                draw(self.format, self.sample_size, filename, stroke)

        return loss

    # def configure_optimizers(self):
    #     optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
    #     # Use a number of steps for the LR scheduler based on an assumed number of steps per epoch
    #     steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
    #     scheduler = CosineAnnealingLR(optimizer, T_max=steps_per_epoch * 10)
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, batch):
        Set, Strokes = batch
        condition, mu, sigma = self.encoder(Set)

        batch_size, num_strokes, stroke_dim = Strokes.shape
        noise = torch.randn_like(Strokes)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=self.device).long()

        Strokes_flat = Strokes.view(batch_size * num_strokes, stroke_dim)
        noise_flat = noise.view(batch_size * num_strokes, stroke_dim)
        timesteps_repeated = timesteps.repeat_interleave(num_strokes)

        noisy_flat = self.noise_scheduler.add_noise(Strokes_flat, noise_flat, timesteps_repeated)
        noisy = noisy_flat.view(batch_size, num_strokes, stroke_dim)

        condition = condition.unsqueeze(1).expand(-1, num_strokes, -1)

        noise_pred = self.decoder(noisy, timesteps.unsqueeze(1).expand(-1, num_strokes), condition)
        loss_mse = F.mse_loss(noise_pred, noise)
        KLD = -0.5 * torch.sum(1 + torch.log(sigma.pow(2) + 1e-8) - mu.pow(2) - sigma.pow(2))
        loss = loss_mse + 0.001 * KLD  # Adjusted KL divergence weight
        return loss
            
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_name = 'Efficient_Run_6'
    format_path = 'format.svg'
    train_path = 'Data/1k.pt'
    val_path = 'Data/1k_val.pt'

    learning_rate = 1e-4
    size = 512
    BATCH_SIZE = 32
    hidden_size = 1024
    samples = 2000
    steps = 1000
    sample_steps = 30
    beta_schedule = 'linear'
    dim_in = 6
    gpu_num = 1

    wand_b_key = 'd049ecc2204afecf6337f395c33813e81fe720e7'
    wandb.login(key=wand_b_key)

    wandb_logger = WandbLogger(name=experiment_name, project='Your Stroke Cloud')

    train_set = Tensor(train_path)
    val_set = Tensor(val_path)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, collate_fn=my_collate, num_workers=4, persistent_workers=True)

    torch.set_float32_matmul_precision("high")

    decoder = CachedMLP(
        hidden_size=hidden_size,
        hidden_layers=3,
        emb_size=64,
        dropout_rate=0.1  # Added dropout
    )

    encoder = SetTransformer(
        dim_input=dim_in,
        num_outputs=1,
        dim_output=256,
        num_inds=16,
        dim_hidden=512,
        num_heads=8,
        ln=True,
        dropout_rate=0.1  # Added dropout
    )

    os.makedirs(f"Results/{experiment_name}", exist_ok=True)
    os.makedirs(f"Models/{experiment_name}", exist_ok=True)

    scheduler = DDPMScheduler(beta_end=1e-4, beta_start=1e-5, num_train_timesteps=steps, beta_schedule=beta_schedule)
    ddim_s = DDIMScheduler(beta_end=1e-4, beta_start=1e-5, num_train_timesteps=steps, beta_schedule=beta_schedule)
    ddim_s.set_timesteps(sample_steps)
    sample_steps = list(range(sample_steps))

    # Define callbacks
    callbacks = [
        StochasticWeightAveraging(swa_lrs=learning_rate),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(monitor='val_loss', patience=10, mode='min')
    ]

    enable_checkpointing = True
    if enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"Models/{experiment_name}/",
            filename="{epoch:02d}-{val_loss:.3f}",
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )
        callbacks.append(checkpoint_callback)

    # Calculate the number of steps per epoch
    steps_per_epoch = len(train_loader)
    
    # Initialize the SRM model with the steps_per_epoch parameter
    srm_model = EnhancedEfficientSRM(encoder, decoder, scheduler, ddim_s, experiment_name, samples, sample_steps, format_path, size, dim_in, learning_rate, steps_per_epoch=steps_per_epoch)

    # Proceed with the trainer setup...
    trainer = L.Trainer(
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        max_epochs=2000,
        check_val_every_n_epoch=5,
        enable_progress_bar=True,
        profiler="simple",
        callbacks=callbacks,
        precision="16-mixed",
        log_every_n_steps=1,
        enable_model_summary=True,
        enable_checkpointing=True,
    )

    trainer.fit(model=srm_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()