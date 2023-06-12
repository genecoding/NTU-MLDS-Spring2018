import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from utils import *


class Discriminator(nn.Module):
    def __init__(self, channels, embed_dim, num_classes):
        super().__init__()
        
        self.label_embedding = nn.Sequential(
            nn.Embedding(num_classes, embed_dim),
            nn.Linear(embed_dim, 64*64),
            )
        
        self.model = nn.Sequential(
            # input (channels+1, 64, 64)
            nn.Conv2d(channels+1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            # output (64, 32, 32)
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # output (128, 16, 16)
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # output (256, 8, 8)
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # output (512, 4, 4)
            
            nn.Conv2d(512, 1, 4, stride=1, padding=0),
            # nn.Sigmoid()
            # output (1, 1, 1)
            )
    
    def forward(self, images, labels):
        label_fmaps = self.label_embedding(labels).reshape(-1, 1, 64, 64)
        input = torch.cat((images, label_fmaps), 1)
        return self.model(input).reshape(-1)

    
class Generator(nn.Module):
    def __init__(self, channels, noise_dim, embed_dim, num_classes):
        super().__init__()
        
        self.label_embedding = nn.Sequential(
            nn.Embedding(num_classes, embed_dim),
            nn.Linear(embed_dim, embed_dim),
            )
        
        self.model = nn.Sequential(
            # input (noise_dim+embed_dim, 1, 1)
            nn.ConvTranspose2d(noise_dim+embed_dim, 512, 4, stride=1, padding=0),
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
            
    def forward(self, noise, labels):
        label_fmaps = self.label_embedding(labels).unsqueeze(-1).unsqueeze(-1)
        input = torch.cat((noise, label_fmaps), 1)
        return self.model(input)


class CGAN(nn.Module):
    """
    CGAN based on WGAN-GP
    """
    def __init__(self, channels=3, noise_dim=100, embed_dim=100, num_classes=119, n_critic=5, lambda_=10):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = SummaryWriter('./log/cgan')
        self.noise_dim = noise_dim
        self.n_critic = n_critic
        self.lambda_ = lambda_
        
        self.D = Discriminator(channels, embed_dim, num_classes)
        self.G = Generator(channels, noise_dim, embed_dim, num_classes)
        self.D.to(self.device)
        self.G.to(self.device)
        # self.D.apply(weights_init)
        # self.G.apply(weights_init)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.9))
        
        print(self.D)
        print(self.G)
    
    def _discriminator_loss(self, logtis_real, logtis_fake):
        return -torch.mean(logtis_real) + torch.mean(logtis_fake)
    
    def _generator_loss(self, logtis_fake):
        return -torch.mean(logtis_fake)
        
    def _gradient_penalty(self, real_images, fake_images, labels):
        batch_size = real_images.shape[0]
        # sample from uniform distribution
        eps = torch.rand(batch_size, 1, 1, 1, device=self.device)
        
        interpolated_imgs = eps * real_images + (1 - eps) * fake_images
        interpolated_imgs.requires_grad_(True)
        logtis_inter = self.D(interpolated_imgs, labels)
        
        gradients = torch.autograd.grad(
            outputs=logtis_inter,
            inputs=interpolated_imgs,
            grad_outputs=torch.ones_like(logtis_inter),
            retain_graph=True,
            create_graph=True,
            )[0]
        gradients = gradients.reshape(batch_size, -1)
        grad_norm = gradients.norm(dim=1)
        
        return torch.mean((grad_norm - 1)**2)

    def train(self, data_loader, save_imgname='cgan_result.jpg', show_every=250, batch_size=64, num_epochs=10):
        iter_count = 0
        for epoch in range(num_epochs):
            for i, (x, y) in enumerate(data_loader):
                if len(x) != batch_size:
                    continue
                
                # =====================    
                #  train discriminator
                # =====================
                self.D_optimizer.zero_grad()
                labels = y.to(self.device)
                real_images = x.to(self.device)
                logits_real = self.D(real_images, labels)
        
                g_fake_seed = torch.randn(batch_size, self.noise_dim, 1, 1, device=self.device)
                fake_images = self.G(g_fake_seed, labels).detach()
                logits_fake = self.D(fake_images, labels)
        
                d_total_error = self._discriminator_loss(logits_real, logits_fake) \
                                + self.lambda_ * self._gradient_penalty(real_images, fake_images, labels)
                d_total_error.backward()        
                self.D_optimizer.step()
    
                # =================
                #  train generator
                # =================
                if iter_count % self.n_critic == 0:
                    self.G_optimizer.zero_grad()
                    g_fake_seed = torch.randn(batch_size, self.noise_dim, 1, 1, device=self.device)
                    fake_images = self.G(g_fake_seed, labels)
                    gen_logits_fake = self.D(fake_images, labels)
                        
                    g_error = self._generator_loss(gen_logits_fake)
                    g_error.backward()
                    self.G_optimizer.step()
                
                # log and save
                if i % 50 == 0:
                    info = {
                        'D': d_total_error.item(), 
                        'G': g_error.item()
                    }
                    self.logger.add_scalars('loss', info, iter_count)
    
                if iter_count % show_every == 0:
                    print('Epoch {}/{} Iter {}\tD_Loss: {:.4}, G_Loss: {:.4}'.format(
                          epoch+1, num_epochs, iter_count, d_total_error.item(), g_error.item()))
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
                
    def test(self, tags, batch_size=25, save_imgname='cgan_gen.jpg'):
        self.G.eval()  # eval mode
        
        with torch.no_grad():
            g_seed = torch.randn(batch_size, self.noise_dim, 1, 1, device=self.device)
            labels = torch.tensor([tag_dict[t] for t in tags]).to(self.device)
            gen_images = self.G(g_seed, labels)
            gen_images = gen_images.data.mul(0.5).add(0.5).cpu()  # denormalize
            show_images(gen_images)
            plt.savefig(save_imgname)
            plt.show()   
        
        self.G.train()  # resume to train mode
    
    def save_model(self, save_path='./saved_model/cgan'):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.D.state_dict(), os.path.join(save_path, 'D_model.pt'))
        torch.save(self.G.state_dict(), os.path.join(save_path, 'G_model.pt'))
        print(f'Save model to {save_path}')
    
    def load_model(self, load_path='./saved_model/cgan'):
        self.D.load_state_dict(torch.load(os.path.join(load_path, 'D_model.pt')))
        self.G.load_state_dict(torch.load(os.path.join(load_path, 'G_model.pt')))
        print(f'Load model from {load_path}')    