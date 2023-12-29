import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels: int = 3, latent_channels: int = 512) -> None:
        super(Encoder, self).__init__()

        self.layer1 = nn.Sequential( 
            nn.Conv2d(input_channels, latent_channels//16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//16),
            nn.Tanh(),
        )
        self.layer2 = nn.Sequential( 
            nn.Conv2d(latent_channels//16, latent_channels//8, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(latent_channels//8),
            nn.Tanh(),
        )
        self.layer3 = nn.Sequential( 
            nn.Conv2d(latent_channels//8, latent_channels//4, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(latent_channels//4),
            nn.Tanh(),
        )
        self.layer4 = nn.Sequential( 
            nn.Conv2d(latent_channels//4, latent_channels//2, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(latent_channels//2),
            nn.Tanh(),
        )
        self.layer5 = nn.Sequential( 
            nn.Conv2d(latent_channels//2, latent_channels, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(latent_channels),
            nn.Tanh(),
        )
        self.conv_mu = nn.Conv2d(
            latent_channels, latent_channels, kernel_size=4, stride=1, padding=0, bias=False)
        self.conv_log_var = nn.Conv2d(
            latent_channels, latent_channels, kernel_size=4, stride=1, padding=0, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x            [B,   3, 128, 128]
        x = self.layer1(x) # [B,  32,  64,  64]
        x = self.layer2(x) # [B,  64,  32,  32]
        x = self.layer3(x) # [B, 128,  16,  16]
        x = self.layer4(x) # [B, 256,   8,   8]
        x = self.layer5(x) # [B, 512,   4,   4]
        
        # Compute mean and logarithm of sd
        # (use logarithm to produce negative values)
        mu = self.conv_mu(x)           # [B, 512, 1, 1]
        log_var = self.conv_log_var(x) # [B, 512, 1, 1]

        return mu, log_var
    

class Decoder(nn.Module):
    def __init__(self, input_channels: int = 3, latent_channels: int = 512) -> None:
        super(Decoder, self).__init__()

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.Tanh(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, latent_channels//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//2),
            nn.Tanh(),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels//2, latent_channels//4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//4),
            nn.Tanh(),
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels//4, latent_channels//8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//8),
            nn.Tanh(),
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels//8, latent_channels//16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels//16),
            nn.Tanh(),
        )
        self.deconv6 = nn.ConvTranspose2d(
            latent_channels//16, input_channels, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # input z:            [B, 512,   1,   1]
        x = self.deconv1(z) # [B, 512,   4,   4]
        x = self.deconv2(x) # [B, 256,   8,   8]
        x = self.deconv3(x) # [B, 128,  16,  16]
        x = self.deconv4(x) # [B,  64,  32,  32]
        x = self.deconv5(x) # [B,  32,  64,  64]
        x = self.deconv6(x) # [B,   3, 128, 128]
        return x

    
class VAE(nn.Module):
    def __init__(self, input_channels: int, latent_channels: int) -> None:
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels=input_channels, latent_channels=latent_channels)
        self.decoder = Decoder(input_channels=input_channels, latent_channels=latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encoder(x)
        # log(var) = log(sd**2)
        # sd**2 = exp(log(var))
        # sd = exp(0.5*log(var))
        z = self.reparameterization_trick(mu=mu, sd=torch.exp(0.5*log_var))
        x_hat = self.decoder(z)
        return x_hat, mu, log_var
    
    def reparameterization_trick(self, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu) # sample from N(0,I)
        z = mu + sd * eps
        return z
    
    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        We want to maximize the quality of reconstruction (minimize MSE).
        """
        mse = torch.sum((x - x_hat)**2) # by batch
        return mse
    
    def kl_divergence(self, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
        """
        We want to minimize KL divergence between the learned distribution 
        of z-space and the prior distribution of z-space.
        """
        kl_distance = 0.5 * torch.sum(1 + torch.log(sd**2) - mu**2 - sd**2) # by batch
        return -kl_distance