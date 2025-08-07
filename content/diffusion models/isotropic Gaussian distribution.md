TLDR: An [[isotropic]] gaussian is one where the covariance matrix is represented by the simplified matrix $Σ=σ^2I$.

Some motivations:

Consider the traditional gaussian distribution:

$$N(μ,Σ)$$

where μ is the mean and Σ is the covariance matrix.

Consider how the number of free parameters in this Gaussian grows as the number of dimensions grows.

μ will have a linear growth. Σ will have a quadratic growth!

<span style="color:red">This quadratic growth can be very computationally expensive, so Σ is often restricted as  </span>$\color{green}Σ=σ^2I$ <span style="color:red">where</span> $\color{green}σ^2I$ <span style="color:red">is a scalar variance multiplied by an identity matrix.</span>

Note that this results in Σ where all dimensions are independent and where the variance of each dimension is the same. So the gaussian will be circular/spherical.