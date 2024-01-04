import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from collections import OrderedDict
import pytorch_lightning as pl
from collections import OrderedDict




# Detective: fake or no fake -> 1 output [0, 1]
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
  
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)
    

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.lin1 = nn.Linear(latent_dim, 64*310*277)  # Adjusted for [3, 1240, 1104]
        self.ct1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # [n, 32, 620, 554]
        self.ct2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # [n, 16, 1240, 1104]
        self.conv = nn.Conv2d(16, 3, kernel_size=7, padding=3)  # [n, 3, 1240, 1104]

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 310, 277)  # Adjusted for [3, 1240, 1104]

        # Upsample (transposed conv) 620x554 (32 feature maps)
        x = self.ct1(x)
        x = F.relu(x)

        # Upsample to 1240x1104 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)

        # Convolution to 1240x1104 (3 feature maps)
        return self.conv(x)





class GAN(pl.LightningModule):
    ## Initialize. Define latent dim, learning rate, and Adam betas 
    def __init__(self, latent_dim=100, lr=0.0002, 
                 b1=0.5, b2=0.999, batch_size=128):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()
        self.validation_z = torch.randn(8, self.hparams.latent_dim)
    
    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        if optimizer_idx == 0:
            self.generated_imgs = self(z)
            predictions = self.discriminator(self.generated_imgs)
            g_loss = self.adversarial_loss(predictions, torch.ones(real_imgs.size(0), 1))
            
            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        if optimizer_idx == 1:
            real_preds = self.discriminator(real_imgs)
            real_loss = self.adversarial_loss(real_preds, torch.ones(real_imgs.size(0), 1))

            fake_preds = self.discriminator(self(z).detach())
            fake_loss = self.adversarial_loss(fake_preds, torch.zeros(real_imgs.size(0), 1)) 

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output
            

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        return opt_g, opt_d
    

    def on_epoch_end(self):
        # log sampled images
        sample_imgs = self(self.validation_z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
    
    

    