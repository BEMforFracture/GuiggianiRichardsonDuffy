# Appendix

## Analytical formulas

### Laplace scalar hypersingular kernel

$$F_{-2}(\theta) = \mathcal V(\mathbf A(\theta)) N^a(\bm \eta) \mathbf J(\bm \eta)=\frac{-1}{4\pi A(\theta)^3}J_i(\hat{\mathbf y})n_i(\bm\eta)$$

With $A(\theta) = \|\mathbf A(\theta)\|$ et $\hat{\mathbf A}(\theta) = \mathbf A(\theta)/A(\theta)$.

$$F_{-1}(\theta) = \frac{3J(\bm\eta)}{4\pi A(\theta)^5}A_m(\theta)B_m(\theta)N^a(\bm\eta) - \frac{J(\bm\eta)}{4\pi A(\theta)^3}c_{\alpha}(\theta)\partial_{\alpha}N^a(\bm\eta) -\\
\frac{1}{4\pi A(\theta)^3}c_{\alpha}(\theta)\partial_{\alpha}J(\bm\eta)N^a(\bm\eta)$$

### Elastostatics hypersingular integral

We let :

$$\bm{\mathcal V}_{ijk\ell}(\mathbf r) = \frac{2\mu\Lambda}{r^3}\{\left\{3\left[(1-2\nu)\hat r_in_k+3\nu\hat r_kn_i\right]\hat{\mathbf r}\cdot \mathbf n^y\right\}+3\left[\nu\delta_{ik} - 5\hat r_i\hat r_k\right](\hat{\mathbf r}\cdot\mathbf n^x)(\hat{\mathbf r}\cdot\mathbf n^y)\\+3\left[(1-2\nu)\hat r_kn^y_i+\nu\hat r_in^y_k\right]\hat{\mathbf r}\cdot\mathbf n^x\\+(\mathbf n^y\cdot\mathbf n^x)\left[3\nu\hat r_i\hat r_k+(1-2\nu)\delta_{ik}\right] + (4\nu-1)n^y_in^x_k + (1-2\nu)n^y_kn^x_i\}$$

So that the elastostatics hypersingular kernel can be written as :

$$V_{ik}(\mathbf y, \mathbf x) = \bm{\mathcal V}_{ijk\ell}(\mathbf r)n_j(\bm x)n_{\ell}(\bm y), \quad \mathbf r = \mathbf y - \mathbf x$$

Where $V_{ik}$ is the tensor kernel resulting from applying twice the elastic traction operator to the Kelvin fundamental solution 

$$V_{ik}(\mathbf y, \mathbf x) = -C_{ijab}\partial_b C_{k\ell cd}\partial_d U_{ac}(\mathbf r), \quad U_{ik}(\mathbf r) = \frac{\Lambda}{2\mu r}\left[\hat r_i\hat r_k + (3-4\nu)\delta_{ik}\right]$$

Then, we have the following analytical formulas for the coefficients of the Laurent's expansion of $F(\rho, \theta)$ :

$$F_{-2}(\theta) = \bm{\mathcal V}_{ijk\ell}(\mathbf A(\theta))J_{\ell}(\bm \eta)n_j(\bm\eta)N^a(\bm \eta)$$