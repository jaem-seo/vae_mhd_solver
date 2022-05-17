# VAE-like MHD equilibrium solver
- Magneto-Hydro-Dynamics (MHD) is the physics of electromagnetically interacting fluids.
- Solving the MHD force balance equation is required first for analyzing fusion plasmas in a tokamak.
- This repository describes a variational auto-encoder (VAE)-like neural network to solve ideal MHD equilibrium.

# Installation
- You can install by
```
$ git clone https://github.com/jaem-seo/vae_mhd_solver.git
$ cd vae_mhd_solver
```

# Try it out
```
$ python predict.py
```
- This solves 0D-2D quantities of the MHD equilibrium, from 1D input profiles (pressure and current density) and the boundary coordinates.
- The below are sample predictions for 2D magnetic flux structure, ψ and φ.
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/168748038-36ac52c3-b2b4-4a9e-98cb-00af03765977.png">
</p>

# Validation
- The input profiles (pressure and current density) are
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/168748534-ae1fa269-90ed-41c9-9c1b-d791f23a2066.png">
</p>

- Then, the 2D outputs are
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/168748830-af80e7c4-6879-4180-a14a-54b4aa9bca00.png">
</p>

# Note
- The current version has a limitation in that the reconstructed outputs are slightly jagged.

# References
- Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint [arXiv:1312.6114 (2013)](https://arxiv.org/pdf/1312.6114.pdf).
- Lütjens, Hinrich, Anders Bondeson, and Olivier Sauter. "The CHEASE code for toroidal MHD equilibria." Computer physics communications [97.3 (1996): 219-260](https://www.sciencedirect.com/science/article/pii/001046559600046X).
