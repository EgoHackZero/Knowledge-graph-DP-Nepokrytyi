![[Autoencoder.png]]
- **Definition**:  Autoencoders have a so-called "bottleneck" layer in between the encoder and decoder. The encoder first encodes an image into a smaller hidden representation called the "bottleneck", and the decoder then decodes that hidden representation back into an actual image. This forces the network to only keep the most important information in the bottleneck layer.
    
- **Training**: The network is trained to minimize the difference between input and reconstructed output (often via mean squared error).
    
- **Variants**:
    
    - Variational Autoencoder (VAE)
        ![[VAE.png]]
        * latent space is the hidden space where your data is represented in a simplified form.
    - Sparse Autoencoder
        
    - Denoising Autoencoder
        
    - Contractive Autoencoder
        
    - Minimum Description Length Autoencoder
        
    - Concrete Autoencoder
        
- **Advantages of depth**: Deep autoencoders capture hierarchical features and can outperform linear methods like PCA for dimensionality reduction.