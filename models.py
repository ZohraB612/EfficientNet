import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L
from positional_embeddings import PositionalEmbedding
from utils import sample, draw, l_sample
from performer_pytorch import SelfAttention
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.cuda.amp import autocast

device = "cuda" if torch.cuda.is_available() else "cpu"

class l_Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor):
        return x + self.act(self.norm(self.ff(x)))

class CachedMLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal", dropout_rate: float = 0.1):
        super(CachedMLP, self).__init__()
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlps = nn.ModuleList([PositionalEmbedding(emb_size, input_emb, scale=25.0) for _ in range(6)])
        concat_size = (7 * emb_size) + 256
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU(), nn.Dropout(dropout_rate)]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_size, 6))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t, y):
        device = x.device
        x = x.to(device)
        t = t.to(device)
        y = y.to(device)

        if x.dim() == 2:
            x = x.unsqueeze(0)

        batch_size, num_samples, _ = x.shape

        x_embs = [mlp(x[:, :, i]) for i, mlp in enumerate(self.input_mlps)]

        if t.dim() == 1:
            t = t.unsqueeze(0).expand(batch_size, -1)
        t_emb = self.time_mlp(t)

        if y.dim() == 2:
            y = y.unsqueeze(1).expand(-1, num_samples, -1)

        x = torch.cat((*x_embs, t_emb, y), dim=-1)
        return self.joint_mlp(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, h, -1).transpose(1, 2), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.attention = MultiHeadAttention(dim_V, heads=num_heads)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln = nn.LayerNorm(dim_V) if ln else nn.Identity()

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        X = torch.cat([Q, K], dim=1)
        H = self.attention(X)
        H = H[:, :Q.size(1), :]  # Only keep the Q part
        return self.ln(Q + self.fc_o(H))

class EfficientSetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, num_inds, dim_hidden, num_heads, ln, dropout_rate: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            EfficientISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln, dropout_rate=dropout_rate),
            EfficientISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, dropout_rate=dropout_rate)
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear_mu = nn.Linear(dim_hidden, dim_output)
        self.linear_sigma = nn.Linear(dim_hidden, dim_output)
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, X):
        encoded = self.enc(X)
        y = self.dec(encoded.mean(dim=1))
        mu = self.linear_mu(y)
        sigma = F.softplus(self.linear_sigma(y))
        z = mu + sigma * self.N.sample(mu.shape).to(mu.device)
        return z, mu, sigma

class EfficientMAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, dropout_rate: float = 0.1):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.attention = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        Q, K, V = map(lambda t: t.transpose(0, 1), (Q, K, V))
        H, _ = self.attention(Q, K, V)
        H = H.transpose(0, 1)
        return self.ln(Q.transpose(0, 1) + self.dropout(self.fc_o(H)))

class EfficientISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, dropout_rate: float = 0.1):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = EfficientMAB(dim_out, dim_in, dim_out, num_heads, ln=ln, dropout_rate=dropout_rate)
        self.mab1 = EfficientMAB(dim_in, dim_out, dim_out, num_heads, ln=ln, dropout_rate=dropout_rate)

    def forward(self, X):
        batch_size = X.size(0)
        H = self.mab0(self.I.repeat(batch_size, 1, 1), X)
        return self.mab1(X, H)

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, num_inds, dim_hidden, num_heads, ln, dropout_rate: float = 0.1):
        super().__init__()
        self.enc = nn.Sequential(
            EfficientISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln, dropout_rate=dropout_rate),
            EfficientISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, dropout_rate=dropout_rate),
            EfficientISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, dropout_rate=dropout_rate)
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.linear_mu = nn.Linear(dim_hidden, dim_output)
        self.linear_sigma = nn.Linear(dim_hidden, dim_output)
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, X):
        encoded = self.enc(X)
        y = self.dec(encoded.mean(dim=1))
        mu = self.linear_mu(y)
        sigma = F.softplus(self.linear_sigma(y))
        z = mu + sigma * self.N.sample(mu.shape).to(mu.device)
        return z, mu, sigma

class srm(L.LightningModule):
    def __init__(self, encoder, decoder, noise_scheduler, noise_scheduler_sample, experiment_name, samples, sample_steps, format_path, sample_size, dim_in, lr):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_sample = noise_scheduler_sample
        self.experiment_name = experiment_name
        self.samples = samples
        self.sample_size = sample_size
        self.sample_steps = sample_steps
        self.format = format_path
        self.learning_rate = lr
        self.dim_in = dim_in
        self.accumulate_grad_batches = 4
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=['encoder', 'decoder', 'noise_scheduler', 'noise_scheduler_sample'])

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        with autocast():
            Set = batch[0]
            condition, mu, sigma = self.encoder(Set)
            Strokes = batch[1]
            noise = torch.randn(Strokes.shape, device=self.device)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (Strokes.shape[1],), device=self.device).long()
            noisy = self.noise_scheduler.add_noise(Strokes, noise, timesteps)
            noise_pred = self.decoder(noisy, timesteps, condition)
            loss_mse = F.mse_loss(noise_pred, noise)
            KLD = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            KLS = 0  # * self.current_epoch
            loss = loss_mse + (KLS * KLD)
        
        self.manual_backward(loss)
        
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt.step()
            opt.zero_grad()
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        condition, mu, sigma = self.encoder(batch[0])
        for i in range(len(condition)):
            filename = f'Results/{self.experiment_name}/{self.current_epoch}_{i}.svg'
            stroke = sample(self.samples, self.sample_steps, self.decoder, self.noise_scheduler_sample, mu[i], self.dim_in)
            draw(self.format, self.sample_size, filename, stroke)
        
        Set = batch[0]
        Strokes = batch[1]
        noise = torch.randn(Strokes.shape, device=self.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (Strokes.shape[1],), device=self.device).long()
        noisy = self.noise_scheduler.add_noise(Strokes, noise, timesteps)
        noise_pred = self.decoder(noisy, timesteps, condition)
        loss_mse = F.mse_loss(noise_pred, noise)
        KLD = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        KLS = 0  # * self.current_epoch
        val_loss = loss_mse + (KLS * KLD)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000], gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def compute_loss(self, batch):
        Set = batch[0]
        condition, mu, sigma = self.encoder(Set)
        Strokes = batch[1]
        noise = torch.randn(Strokes.shape, device=self.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (Strokes.shape[1],), device=self.device).long()
        noisy = self.noise_scheduler.add_noise(Strokes, noise, timesteps)
        
        # Ensure condition has the correct shape
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)
        
        noise_pred = self.decoder(noisy, timesteps, condition)
        loss_mse = F.mse_loss(noise_pred, noise)
        KLD = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        KLS = 0  # * self.current_epoch
        loss = loss_mse + (KLS * KLD)
        return loss

