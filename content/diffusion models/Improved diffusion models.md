Original paper [link](https://openreview.net/pdf?id=-NEXDKk8gZ)

---

## 1. Innovations in **Improved DDPM** (Nichol & Dhariwal, 2021)

### **A. Learnable Reverse Variances**

- Instead of fixing $\sigma_t^2$​ from the forward process, they **parameterize and learn** it.
- They model
$$
\log \sigma_t^2 = v \cdot \log \beta_t + (1-v) \cdot \log \tilde{\beta}_t
$$
where $v$ is predicted by the network and interpolates between the forward variance $\beta_t$​ and the “posterior variance” $\tilde{\beta}_t$​.

- **Mathematical benefit:** reduces variance mismatch between forward and reverse processes, improving sample quality and likelihood.    

---

### **B. Hybrid Loss – Weighted Combination**

- They move beyond the _pure noise prediction loss_ and combine:
    
    1. **Noise-prediction MSE loss** (as in DDPM).
        
    2. **Variational bound term** from the ELBO for variance learning.
        
- Final loss: