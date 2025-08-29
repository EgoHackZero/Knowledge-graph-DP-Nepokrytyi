Original paper [link](https://openreview.net/pdf?id=-NEXDKk8gZ)

---

## 2. Innovations in **Improved DDPM** (Nichol & Dhariwal, 2021)

### **A. Learnable Reverse Variances**

- Instead of fixing $\sigma_t^2$​ from the forward process, they **parameterize and learn** it.
- They model

$$
\log \sigma_t^2 = v \cdot \log \beta_t + (1-v) \cdot \log \tilde{\beta}_t
$$​
	where $v$ is predicted by the network and interpolates between the forward variance $\beta_t$​ and the “posterior variance” $\tilde{\beta}_t$​.
- **Mathematical benefit:** reduces variance mismatch between forward and reverse processes, improving sample quality and likelihood.    

---

### **B. Hybrid Loss – Weighted Combination**

- They move beyond the _pure noise prediction loss_ and combine:
    
    1. **Noise-prediction MSE loss** (as in DDPM).
        
    2. **Variational bound term** from the ELBO for variance learning.
        
- Final loss:   

$$
L = L_{\text{simple}} + \lambda L_{\text{vb}}
$$





























$$
L_{\text{vb}} = \mathrm{KL}\big(q(x_{t-1} \mid x_t, x_0) \ \|\ p_\theta(x_{t-1} \mid x_t) \big)
$$
    where $L_{\text{vb}}$​ is the [[KL term]] between the learned reverse distribution and the true posterior $q(x_{t-1} \mid x_t, x_0)$.

---

### **C. Cosine Noise Schedule**

- Instead of the linear $\beta_t$​ schedule in DDPM, they use:

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left( \frac{t/T + s}{1+s} \cdot \frac{\pi}{2} \right)
$$
    with small offset $s \approx 0.008$.
- This concentrates noise addition in later steps, improving training stability and sample quality.

---

### **D. Classifier-Free Guidance (for conditional tasks)**

- They show that guidance can be done without an external classifier, by training the model with randomly dropped conditioning information and interpolating predictions at inference.
## 3. Summary Table

|Change|Original DDPM|Improved DDPM|
|---|---|---|
|Reverse variance|Fixed from forward process|Learned (interpolated between forward & posterior variance)|
|Loss|Only noise-prediction MSE|Hybrid loss: noise MSE + KL variance term|
|Noise schedule|Linear βt\beta_tβt​|Cosine βt\beta_tβt​ schedule|
|Guidance|Requires classifier|Classifier-free guidance|
