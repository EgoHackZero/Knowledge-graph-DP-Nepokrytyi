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

Let $q(x_0)$ be the real data distribution, say of "real images". We can sample from this distribution to get an image, $x_0∼q(x_0)$. We define the forward diffusion process $q(x_t∣x_{t−1})$ which adds Gaussian noise at each time step $t$, according to a known variance schedule $0<β_1<β_2<...<β_T<1$ as

$$
q(x_t∣x_{t−1})=N(x_t;\sqrt{1−β_t} \cdot x_{t−1},β_tI).
$$

Recall that a normal distribution (also called Gaussian distribution) is defined by 2 parameters: a mean $μ$ and a variance $σ^2≥0$. Basically, each new (slightly noisier) image at time step $t$ is drawn from a **conditional Gaussian distribution** with $μ_t=\sqrt{1−β_t}\cdot x_{t−1}$​ and $σ_t^2=β_t$​, which we can do by sampling $ϵ∼N(0,I)$ and then setting $x_t=\sqrt{1−β_t}\cdot x_{t−1}+\sqrt{β_t}ϵ$.

Note that the $β_t$​ aren't constant at each time step $t$ (hence the subscript) --- in fact one defines a so-called **"variance schedule"**, which can be linear, quadratic, cosine, etc. as we will see further (a bit like a learning rate schedule).

Now we're going to leverage a neural network to **approximate (learn) this conditional probability distribution**, let's call it $p_θ(x_{t−1}∣x_t)$, with $θ$ being the parameters of the neural network, updated by gradient descent. If we assume this reverse process is Gaussian as well, then recall that any Gaussian distribution is defined by 2 parameters:

- a mean parametrized by $μ_θ$​;
- a variance parametrized by $Σ_θ$;
​
so we can parametrize the process as

$$
p_θ(x_{t−1}∣x_t)=N(x_{t−1};μ_θ(x_t,t),Σ_θ(x_t,t))
$$

where the mean and variance are also conditioned on the noise level $t$. 

At the first paper author decided to **keep the variance fixed, and let the neural network only learn (represent) the mean $μ_θ$​ of this conditional probability distribution**. From the paper:

> First, we set $Σ_θ​(x_t​,t)=σ_t^2​I$ to untrained time dependent constants. Experimentally, both $σ_t^2=β_t$ and $σ_t^2=\tilde{\beta}_t$​ (see paper) had similar results.

- **Loss function:** simplified to an MSE between predicted noise and true noise: 

$$
L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \big[ \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \|^2 \big].
$$

This was then later improved in the [[Improved diffusion models]] paper, where a neural network also learns the variance of this backwards process, besides the mean.

# Defining an objective function (by reparametrizing the mean)

$$
q(x_t \mid x_0) = \mathcal{N}\big(x_t; \sqrt{\bar{\alpha}_t}x_0,\,(1-\bar{\alpha}_t)I \big)
$$

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}
\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \, \epsilon_\theta(x_t, t)\right)
$$

$$
\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2
= \lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\, t\big) \rVert^2
$$


![[Pasted image 20250829010726.png]]

- Most diffusion models use architectures that are some variant of a [U-net](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F1505.04597) and that's what we'll use here.
	- a [[ResNet downsampling]] block with [[spatial self-attention]]

