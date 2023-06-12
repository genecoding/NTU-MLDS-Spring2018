import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from utils import *


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.model = nn.Sequential(
            # input (channels, 64, 64)
            nn.Conv2d(channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            # output (64, 32, 32)
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # output (128, 16, 16)
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # output (256, 8, 8)
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # output (512, 4, 4)
            
            nn.Conv2d(512, 1, 4, stride=1, padding=0),
            # nn.Sigmoid()
            # output (1, 1, 1)
            )
    
    def forward(self, input):
        return self.model(input).reshape(-1)

    
class Generator(nn.Module):
    def __init__(self, channels, noise_dim):
        super().__init__()
        
        self.model = nn.Sequential(
            # input (noise_dim, 1, 1)
            nn.ConvTranspose2d(noise_dim, 512, 4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # output (512, 4, 4)
            
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # output (256, 8, 8)
            
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # output (128, 16, 16)
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # output (64, 32, 32)
            
            nn.ConvTranspose2d(64, channels, 4, stride=2, padding=1),
            nn.Tanh()
            # output (channels, 64, 64)
            )
            
    def forward(self, input):
        return self.model(input)


class WGAN(nn.Module):
    def __init__(self, channels=3, noise_dim=100, n_critic=5, clip_thres=0.01):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = SummaryWriter('./log/wgan')
        self.noise_dim = noise_dim
        self.n_critic = n_critic
        self.clip_thres = clip_thres
        
        self.C = Discriminator(channels)  # critic
        self.G = Generator(channels, noise_dim)
        self.C.to(self.device)
        self.G.to(self.device)
        self.C.apply(weights_init)
        self.G.apply(weights_init)
        self.C_optimizer = optim.RMSprop(self.C.parameters(), lr=5e-5)
        self.G_optimizer = optim.RMSprop(self.G.parameters(), lr=5e-5)
        
        print(self.C)
        print(self.G)
    
    def _critic_loss(self, logtis_real, logtis_fake):
        return -torch.mean(logtis_real) + torch.mean(logtis_fake)
    
    def _generator_loss(self, logtis_fake):
        return -torch.mean(logtis_fake)

    def train(self, data_loader, save_imgname='wgan_result.jpg', show_every=250, batch_size=64, num_epochs=10):
        iter_count = 0
        for epoch in range(num_epochs):
            for i, (x, _) in enumerate(data_loader):
                if len(x) != batch_size:
                    continue
                
                # ==============   
                #  train critic
                # ==============
                self.C_optimizer.zero_grad()
                real_images = x.to(self.device)
                logits_real = self.C(real_images)
        
                g_fake_seed = torch.randn(batch_size, self.noise_dim, 1, 1, device=self.device)
                fake_images = self.G(g_fake_seed).detach()
                logits_fake = self.C(fake_images)
        
                c_total_error = self._critic_loss(logits_real, logits_fake)
                c_total_error.backward()        
                self.C_optimizer.step()
                    
                # weight clippling
                for p in self.C.parameters():
                    p.data.clamp_(-self.clip_thres, self.clip_thres)
    
                # =================
                #  train generator
                # =================
                if iter_count % self.n_critic == 0:
                    self.G_optimizer.zero_grad()
                    g_fake_seed = torch.randn(batch_size, self.noise_dim, 1, 1, device=self.device)
                    fake_images = self.G(g_fake_seed)
                    gen_logits_fake = self.C(fake_images)
                        
                    g_error = self._generator_loss(gen_logits_fake)
                    g_error.backward()
                    self.G_optimizer.step()
                
                # log and save
                if i % 50 == 0:
                    info = {
                        'C': c_total_error.item(), 
                        'G': g_error.item()
                    }
                    self.logger.add_scalars('loss', info, iter_count)
    
                if iter_count % show_every == 0:
                    print('Epoch {}/{} Iter {}\tC_Loss: {:.4}, G_Loss: {:.4}'.format(
                          epoch+1, num_epochs, iter_count, c_total_error.item(), g_error.item()))
                    gen_images = fake_images.data.mul(0.5).add(0.5).cpu()  # denormalize
                    show_images(gen_images[0:25])
                    plt.show()
                    self.save_model()
                    print()
                iter_count += 1
                
            if epoch == num_epochs - 1:
                show_images(gen_images[0:25])
                plt.savefig(save_imgname)
                self.save_model()
                
    def test(self, batch_size=25, save_imgname='wgan_gen.jpg'):
        self.G.eval()  # eval mode
        
        with torch.no_grad():
            g_seed = torch.randn(batch_size, self.noise_dim, 1, 1, device=self.device)
            gen_images = self.G(g_seed)
            gen_images = gen_images.data.mul(0.5).add(0.5).cpu()  # denormalize
            show_images(gen_images)
            plt.savefig(save_imgname)
            plt.show()
            
        self.G.train()  # resume to train mode
    
    def save_model(self, save_path='./saved_model/wgan'):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.C.state_dict(), os.path.join(save_path, 'C_model.pt'))
        torch.save(self.G.state_dict(), os.path.join(save_path, 'G_model.pt'))
        print(f'Save model to {save_path}')
    
    def load_model(self, load_path='./saved_model/wgan'):
        self.C.load_state_dict(torch.load(os.path.join(load_path, 'C_model.pt')))
        self.G.load_state_dict(torch.load(os.path.join(load_path, 'G_model.pt')))
        print(f'Load model from {load_path}')    