(also known as DDPMs, diffusion models, score-based generative models or simply [autoencoders](https://benanne.github.io/2022/01/31/diffusion.html))

So it all starts with DDPM paper by ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) but the idea of diffusion for generative modeling was actually already introduced in ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)). However, it took until ([Song et al., 2019](https://arxiv.org/abs/1907.05600)) (at Stanford University), and then ([Ho et al., 2020](https://arxiv.org/abs/2006.11239)) (at Google Brain) who independently improved the approach.

Note that there are [[several perspectives on diffusion models]].

# What is a diffusion model?

A **diffusion probabilistic model** is a parameterized Markov chain trained using variational inference to produce samples matching the data after finite time. 
![[diffusion_figure.png]]
- Most diffusion models use architectures that are some variant of a [U-net](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F1505.04597) and that's what we'll use here.
	- a [[ResNet downsampling]] block with [[spatial self-attention]]
