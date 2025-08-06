(also known as DDPMs, diffusion models, score-based generative models or simply [autoencoders](https://benanne.github.io/2022/01/31/diffusion.html))

So it all starts with DDPM paper by ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) but the idea of diffusion for generative modeling was actually already introduced in ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)). However, it took until ([Song et al., 2019](https://arxiv.org/abs/1907.05600)) (at Stanford University), and then ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) (at Google Brain) who independently improved the approach.

Note that there are [[several perspectives on diffusion models]].

# What is a diffusion model?

A **diffusion probabilistic model** is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time. 

Most often, we think of diffusion models as (**denoising**) diffusion models where **a neural network learns to gradually denoise data** starting from pure noise.

In a bit more detail for images, the set-up consists of 2 processes:

- a fixed (or predefined) [[forward diffusion]] process $q$ of our choosing, that gradually adds Gaussian noise to an image, until you end up with pure noise
- a learned reverse denoising diffusion process $p_θ$​, where a neural network is trained to gradually denoise an image starting from pure noise, until you end up with an actual image.

![[diffusion_figure.png]]
# In more mathematical form

- Most diffusion models use architectures that are some variant of a [U-net](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F1505.04597) and that's what we'll use here.
	- a [[ResNet downsampling]] block with [[spatial self-attention]]

