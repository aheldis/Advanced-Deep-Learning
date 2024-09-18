import torch
from torch.autograd.grad_mode import no_grad
import torch.nn as nn
import torch.nn.functional as F
from ResUNet import ConditionalUnet
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionalDDPM(nn.Module):
    def __init__(self, dmconfig):
        super().__init__()
        self.dmconfig = dmconfig
        self.loss_fn = nn.MSELoss()
        self.network = ConditionalUnet(1, self.dmconfig.num_feat, self.dmconfig.num_classes)

    def scheduler(self, t_s):
        beta_1, beta_T, T = self.dmconfig.beta_1, self.dmconfig.beta_T, self.dmconfig.T
        # ==================================================== #
        #   Inputs:
        #       t_s: the input time steps, with shape (B,1). 
        #   Outputs:
        #       one dictionary containing the variance schedule
        #       $\beta_t$ along with other potentially useful constants.       

        ratio_ts = (t_s - 1) / (T - 1)      
        beta_t = beta_1 + ratio_ts * (beta_T - beta_1)
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        oneover_sqrt_alpha = 1 / torch.sqrt(alpha_t)

        # register a constant beta_Ts for easy computation
        Ts = torch.arange(1, T+1, device=device)
        betas = beta_1 + (Ts - 1) / (T - 1) * (beta_T - beta_1)
        alphas = torch.cumprod(1 - betas, dim=0)

        alpha_t_bar = alphas[t_s - 1]
        sqrt_alpha_bar = torch.sqrt(alpha_t_bar)
        sqrt_oneminus_alpha_bar = torch.sqrt(1 - alpha_t_bar)

        # ==================================================== #
        return {
            'beta_t': beta_t,
            'sqrt_beta_t': sqrt_beta_t,
            'alpha_t': alpha_t,
            'sqrt_alpha_bar': sqrt_alpha_bar,
            'oneover_sqrt_alpha': oneover_sqrt_alpha,
            'alpha_t_bar': alpha_t_bar,
            'sqrt_oneminus_alpha_bar': sqrt_oneminus_alpha_bar
        }

    def forward(self, images, conditions):
        T = self.dmconfig.T
        noise_loss = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given training algorithm.
        #   Inputs:
        #       images: real images from the dataset, with size (B,1,28,28).
        #       conditions: condition labels, with size (B). You should
        #                   convert it to one-hot encoded labels with size (B,10)
        #                   before making it as the input of the denoising network.
        #   Outputs:
        #       noise_loss: loss computed by the self.loss_fn function  .  

        # mask the conditions
        mask_p = self.dmconfig.mask_p
        onehot_cond = F.one_hot(conditions, num_classes=self.dmconfig.num_classes).float()              # (B, 10)
        mask = torch.rand(conditions.size(0), 1, device=device).repeat(1, onehot_cond.size(1))           # (B, 10)                                                         # (B, 1)
        mask_cond = torch.where(mask < mask_p, 
                                torch.full_like(onehot_cond, self.dmconfig.condition_mask_value),
                                onehot_cond)                                                            # (B, 10)

        # sample the timestep from {1, ..., T}
        ts = torch.randint(1, T + 1, size=(images.size(0), 1), device=device)            # (B, 1)

        # sample the Gaussian noise
        noise = torch.randn_like(images, device=device)                                                # (B, C, H, W)

        # corrupt the images
        corrupt_scheduler = self.scheduler(ts)
        corrupted_images = corrupt_scheduler["sqrt_alpha_bar"].view(-1, 1, 1, 1) * images + corrupt_scheduler["sqrt_oneminus_alpha_bar"].view(-1, 1, 1, 1) * noise
        # normalize the timesteps before sending into the network
        ts = (ts - 1) / (T - 1)
        pred_noise = self.network(corrupted_images, ts.view(-1, 1, 1, 1), mask_cond)
        
        noise_loss = self.loss_fn(pred_noise, noise)

        # ==================================================== #
        
        return noise_loss

    def sample(self, conditions, omega):
        T = self.dmconfig.T
        X_t = None
        # ==================================================== #
        # YOUR CODE HERE:
        #   Complete the training forward process based on the
        #   given sampling algorithm.
        #   Inputs:
        #       conditions: encoded labels with size (B,10)
        #       omega: conditional guidance weight.
        #   Outputs:
        #       generated_images  

        batch_size = conditions.size(0)
        # initialize as the random noise
        X_t = torch.randn(batch_size, self.dmconfig.num_channels, self.dmconfig.input_dim[0], self.dmconfig.input_dim[1], device=device)

        with torch.no_grad():
            for t in range(T, 0, -1):
                # sample from the random Gaussian
                z = torch.randn_like(X_t, device=device) if t > 1 else torch.zeros_like(X_t, device=device)

                nts = (t - 1) / (T - 1)
                nts = nts * torch.ones(batch_size, 1, dtype=torch.float, device=device)                                           # normalize and reshape
                no_cond = torch.full(conditions.size(), self.dmconfig.condition_mask_value, device=device)                              # no condition info, all -1
                corrected_noise = (1 + omega) * self.network(X_t, nts, conditions) - omega * self.network(X_t, nts, no_cond)
                
                scheduler_t = self.scheduler(t * torch.ones(batch_size, 1, dtype=torch.int, device=device))
                oneover_sqrt_alpha = scheduler_t["oneover_sqrt_alpha"].view(-1, 1, 1, 1)
                oneminus_alpha_t = 1 - scheduler_t["alpha_t"].view(-1, 1, 1, 1)
                sqrt_oneminus_alpha_bar = scheduler_t["sqrt_oneminus_alpha_bar"].view(-1, 1, 1, 1)
                sqrt_beta_t = scheduler_t["sqrt_beta_t"].view(-1, 1, 1, 1)
                X_t = oneover_sqrt_alpha * (X_t - oneminus_alpha_t / sqrt_oneminus_alpha_bar * corrected_noise) + sqrt_beta_t * z

        # ==================================================== #
        generated_images = (X_t * 0.3081 + 0.1307).clamp(0,1) # denormalize the output images
        return generated_images
        