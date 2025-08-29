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

1. **Variational Objective via ELBO**  
    The forward diffusion process $q$ and the learned reverse process $p_\theta$ together form a structure analogous to a _variational autoencoder_ (VAE). Training leverages the evidence lower bound (ELBO) to minimize the negative log‑likelihood of the original data sample $x_0$​. The total loss decomposes across all diffusion time steps:  
    $$
    L = L_0 + L_1 + \dots + L_T
    $$
    where, except for $L_0$​, each $L_t$​ corresponds to the **KL divergence between two Gaussians**—and simplifies to a squared‑error (L2) term concerning their means.
    
2. **Sampling $x_t$​ Directly ("Nice Property")**  
    Because sums of Gaussians remain Gaussian, it's possible to sample the noisy version $x_t$​ at any time-step $t$ in closed form as:  
    $$
    q(x_t \mid x_0) = \mathcal{N}(x_t; \bar{\alpha}_t x_0,\ (1 - \bar{\alpha}_t)\,I)
    $$
    where $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$. This allows direct sampling of $x_t$​ from $x_0$​, enabling random $t$-step loss evaluation during training.
    
3. **Reparametrizing the Mean: Learning to Predict Noise**  
    Instead of having the network predict the mean $\mu_\theta(x_t, t)$ of the reverse distribution directly, it’s rewritten so the network $\epsilon_\theta(x_t, t)$ predicts the added noise. The mean can then be reconstructed as a function of this prediction. 
    $$
	\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}
	\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \, \epsilon_\theta(x_t, t)\right)
	$$
    The training objective simplifies to minimizing the MSE:  
    $$
	\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2
	= \lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\, t\big) \rVert^2
	$$
    where $x_t$​ is obtained via the "nice property" above and $\epsilon$ is the true noise added.
    
4. **Resulting Training Process**  
	![[Pasted image 20250829010726.png]]In other words:

	- we take a random sample $x_0$​ from the real unknown and possibily complex data distribution $q(x_0)$
	- we sample a noise level $t$ uniformly between $1$ and $T$ (i.e., a random time step)
	- we sample some noise from a Gaussian distribution and corrupt the input by this noise at level $t$ (using the nice property defined above)
	- the neural network is trained to predict this noise based on the corrupted image $x_t$​ (i.e. noise applied on $x_0$​ based on known schedule $β_t$​)

---

### Why This Matters

- **Simplifies optimization**: Using the ELBO lets the model optimize each timestep separately via a straightforward L2 loss.
    
- **Efficiency**: The "nice property" avoids explicitly chaining the forward diffusion to reach $x_t$​.
    
- **Robust learning target**: Predicting noise instead of the mean tends to be more stable and effective during training.


---

- Most diffusion models use architectures that are some variant of a [U-net](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F1505.04597) and that's what we'll use here.
	- a [[ResNet downsampling]] block with [[spatial self-attention]]

