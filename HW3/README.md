# Homework 3
* [3-1 Image Generation]
* [3-2 Text-to-Image Generation]

## Note
* Implement
  - [x] DCGAN (Deep Convolutional GAN)
  - [x] WGAN
  - [x] WGAN-GP
  - [x] CGAN (Conditional GAN)
* Training tips, apply some of [Tips and tricks to make GANs work]
  * Normalize the inputs
  * A modified loss function
  * Use a spherical Z (Dont sample from a Uniform distribution, sample from a gaussian distribution.)
  * BatchNorm
  * Avoid Sparse Gradients: ReLU, MaxPool
  * Use the ADAM Optimizer
  * Train discriminator more (sometimes)
  * Discrete variables in Conditional GANs
* DCGAN
  * Guidelines from [the paper]
    * Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
    * Use batchnorm in both the generator and the discriminator.
    * Remove fully connected hidden layers for deeper architectures.
    * Use ReLU activation in generator for all layers except for the output, which uses Tanh.
    * Use LeakyReLU activation in the discriminator for all layers.
    * Directly applying batchnorm to all layers however, resulted in sample oscillation and model instability. 
      This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer.
  * Weight initialization is necessary.
* Changes from DCGAN to WGAN
  * Rename discriminator to critic.
  * Use weight clipping to enforce 1-Lipschitz continuity on the critic.
  * Change the critic’s activation function from sigmoid to linear.
  * Update the critic more frequently than the generator.
  * No more image labels for the critic.
  * Use Wasserstein loss instead of Binary Crossentropy (BCE) loss.
  * Use RMSProp optimizer.
  * Weight initialization is necessary.
* Changes from WGAN to WGAN-GP
  * Remove batch norm from the critic’s architecture.
  * Use gradient penalty instead of weight clipping to enforce the Lipschitz constraint.
  * Use Adam optimizer (α = 0.0001, β1 = 0.5, β2 = 0.9) instead of RMSProp.
  * Weight initialization is not necessary.

## Result
### 3-1 Image Generation
| DCGAN         |
|---------------|
|![gan_original]|
### 3-2 Text-to-Image Generation
| CGAN           |
|----------------|
|![cgan_original]|

## Reference
* [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks][p1] Alec R. et al
* https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
* [Wasserstein GAN][p2] Martin A. et al
* https://github.com/martinarjovsky/WassersteinGAN
* [Improved Training of Wasserstein GANs][p3] Ishaan G. et al
* https://github.com/igul222/improved_wgan_training



[3-1 Image Generation]: https://docs.google.com/presentation/d/1UdLXHcu-pvvYkNvZIWT7tFbuGO2HzHuAZhcA0Xdrtd8
[3-2 Text-to-Image Generation]: https://docs.google.com/presentation/d/1P5ToVdC_FaFzqC-wD6al6RoLseOgzoyaYESyJasef2E
[Tips and tricks to make GANs work]: https://github.com/soumith/ganhacks
[the paper]: https://arxiv.org/pdf/1511.06434.pdf
[gan_original]: samples/gan_original.png
[cgan_original]: samples/cgan_original.png
[p1]: https://arxiv.org/abs/1511.06434
[p2]: https://arxiv.org/abs/1701.07875
[p3]: https://arxiv.org/abs/1704.00028
