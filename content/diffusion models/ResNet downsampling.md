In classic ResNet (Residual Network), a **downsampling block** (used when reducing spatial resolution and increasing channels) looks like this:

```
Input
 ↓
1x1 Conv (reduce channels)
 ↓
3x3 Conv (stride=2 for downsampling)
 ↓
1x1 Conv (expand channels)
 ↓
Shortcut (with 1x1 Conv + stride=2)
 ↓
Add + ReLU
```

- **Purpose**: Reduce spatial dimensions (e.g., from 56×56 to 28×28) while maintaining representational power.    
- **Residual connection**: Helps with gradient flow during training.