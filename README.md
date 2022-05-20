# VAE-like MHD equilibrium solver
- Magneto-Hydro-Dynamics (MHD) is the physics of electromagnetically interacting fluids.
- Solving the MHD force balance equation is the first step for analyzing fusion plasmas in a tokamak.
- This repository describes a variational auto-encoder (VAE)-like neural network to solve ideal MHD equilibrium.
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/168963839-3114ff3d-b50a-4e6f-aab1-b1d4bd96aea8.png">
</p>

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

# Description
- The input profiles (pressure, current density, and plasma boundary) are
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/168934018-e295b894-6277-47e4-a496-0f18d59ee977.png">
</p>

- Then, the 2D outputs (ground truth and prediction) are
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/168748830-af80e7c4-6879-4180-a14a-54b4aa9bca00.png">
</p>

- Other 0D and 1D physical quantities are also calculated.

# Note
- The current version has a limitation in that the reconstructed outputs are slightly jagged.
- The model is similar to VAE, but the input and output are only physically consistent with each other, not the same structure.
- A simple physical constraint has been added to the loss calculation.
- The physical quantities for input/outputs are normalized according to [CHEASE convention](https://crppwww.epfl.ch/~sauter/chease/chease_normalization.pdf).

# References
- Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint [arXiv:1312.6114 (2013)](https://arxiv.org/abs/1312.6114).
- Miller, R. L., et al. "Noncircular, finite aspect ratio, local equilibrium model." Physics of Plasmas [5.4 (1998): 973-978](https://aip.scitation.org/doi/abs/10.1063/1.872666).
- Lütjens, Hinrich, Anders Bondeson, and Olivier Sauter. "The CHEASE code for toroidal MHD equilibria." Computer physics communications [97.3 (1996): 219-260](https://www.sciencedirect.com/science/article/pii/001046559600046X).

# TODO
- Physics (Grad-Shafranov equation)-informed loss applied
- Additional encoder for general boundary coordinates
