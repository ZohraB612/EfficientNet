import wandb
from Data_Set import my_collate, Tensor
from models import SetTransformer, MLP, srm
import torch
from torch.utils.data import DataLoader
import os
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from diffusers import DDIMScheduler, DDPMScheduler
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint

device = "cuda" if torch.cuda.is_available() else "cpu"

# Updated Experiment Details
experiment_name = 'Experiment-cactus-2'
learning_rate = 3e-4 # was 3e-4
BATCH_SIZE = 64
hidden_size = 2048
num_heads = 8  # Updated number of heads
num_layers = 6
num_inds = 32
beta_end = 1e-4
beta_start = 1e-5
steps = 200
sample_steps = 16  # Updated sample steps used to be 16
dim_in = 6
gpu_num = 1

wand_b_key = 'd049ecc2204afecf6337f395c33813e81fe720e7'
wandb.login(key=wand_b_key)
wandb_logger = WandbLogger(name=experiment_name, project='Your Stroke Cloud')

train_path = 'test_dataset/arr_tensor.pt'
val_path = 'test_dataset/arr_tensor_val.pt'
train_set = Tensor(train_path)
val_set = Tensor(val_path)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, collate_fn=my_collate, pin_memory=True)
val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, collate_fn=my_collate)
torch.set_float32_matmul_precision("medium")

checkpoint_callback = ModelCheckpoint(
    dirpath="Models/{}/".format(experiment_name),
    filename="{epoch:02d}-{global_step}",
)
decoder = MLP(
    hidden_size=hidden_size,
    hidden_layers=num_layers,  # Updated layers
    emb_size=64,
    time_emb="sinusoidal",
    input_emb="sinusoidal"
)

encoder = SetTransformer(
    dim_input=dim_in,
    num_outputs=1,
    dim_output=256,
    num_inds=num_inds,  # Updated number of inducing points
    dim_hidden=256,
    num_heads=num_heads,  # Updated number of heads
    ln=True
)

if not os.path.exists("Results/{}".format(experiment_name)):
    os.makedirs("Results/{}".format(experiment_name))

if not os.path.exists("Models/{}".format(experiment_name)):
    os.makedirs("Models/{}".format(experiment_name))

scheduler = DDPMScheduler(beta_end=beta_end, beta_start=beta_start, num_train_timesteps=steps, beta_schedule='linear')
ddim_s = DDIMScheduler(beta_end=beta_end, beta_start=beta_start, num_train_timesteps=steps, beta_schedule='linear')
ddim_s.set_timesteps(sample_steps)
sample_steps = list(range(sample_steps))

format_path = 'format.svg'
size = 512

srm_model = srm(encoder, decoder, scheduler, ddim_s, experiment_name, 1000, sample_steps, format_path, size, dim_in, learning_rate)

trainer = L.Trainer(
    accelerator='gpu',
    devices=gpu_num,
    strategy='auto',
    logger=wandb_logger,
    max_epochs=-1,
    check_val_every_n_epoch=100,
    enable_progress_bar=True,
    profiler="simple",
    callbacks=[StochasticWeightAveraging(swa_lrs=learning_rate), checkpoint_callback],
    benchmark=True
)

trainer.fit(model=srm_model, train_dataloaders=train_loader, val_dataloaders=val_loader)