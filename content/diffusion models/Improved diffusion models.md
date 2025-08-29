Original paper [link](https://openreview.net/pdf?id=-NEXDKk8gZ)

---

## 2. Innovations in **Improved DDPM** (Nichol & Dhariwal, 2021)

### **A. Learnable Reverse Variances**

- Instead of fixing $\sigma_t^2$​ from the forward process, they **parameterize and learn** it.
- They model

$$
\log \sigma_t^2 = v \cdot \log \beta_t + (1-v) \cdot \log \tilde{\beta}_t
$$