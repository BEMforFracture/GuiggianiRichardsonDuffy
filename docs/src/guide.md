# Guide

## Laurent's expansions methods

The idea is to transform the original integral into a polar integral whose singularity is easier to handle. The elementary integral is written as a finite part integral :

$$\text{f.p.}\int_{\tau_e}K(\mathbf y, \mathbf x)\varphi(\mathbf x) dS_y=\text{f.p.}\int_{\Delta_e}K(\bm \eta, \bm \xi)\varphi(\bm \xi) J(\bm \xi) d\xi$$

Which is then transformed into a polar integral around the singularity point via the change of variables :

$$\hat{\mathbf {x}}=\bm \eta+ \rho \mathbf{c}(\theta), \quad \mathbf c(\theta) = \begin{pmatrix}\cos \theta \\ \sin \theta \end{pmatrix}$$

So that the integral can be reduced, after rearranging terms (see [guiggianiGeneralAlgorithmNumerical1992](@cite)):

$$\int_{0}^{2\pi} \int_{0}^{\hat{\rho}(\theta)}\left\{F(\rho, \theta) - \frac{F_{-2}(\theta)}{\rho^2} - \frac{F_{-1}(\theta)}{\rho}\right\}\mathrm d\rho d\theta + \int_{0}^{2\pi}\left\{F_{-1}(\theta)\log\hat\rho(\theta) - \frac{F_{-2}(\theta)}{\hat\rho(\theta)}\mathrm d\theta\right\}$$

Where $F(\rho, \theta) = K(\bm \eta(\rho, \theta), \bm \xi(\rho, \theta))N^a(\bm \xi(\rho, \theta)) J(\bm \xi(\rho, \theta)) \rho$ ; $N^a$ is the $a$-th shape function associated to the $a$-th node of the element, and $F_{-1}(\theta)$, $F_{-2}(\theta)$ are the coefficients of the Laurent's expansion of $F(\rho, \theta)$ around $\rho = 0$, such that :

$$F(\rho, \theta) = \frac{F_{-2}(\theta)}{\rho^2} + \frac{F_{-1}(\theta)}{\rho} + \mathcal O(1)$$

### Expansion using analytical formula

If the kernel is known analytically, there is not fundamental difficulty in deriving the coefficients $F_{-1}(\theta)$ and $F_{-2}(\theta)$ analytically. This is the approach taken in the original paper by Guiggiani et al. [guiggianiGeneralAlgorithmNumerical1992](@cite) for the Laplace kernel ; we recall it in the next section for the hypersingular integral.

In the case translation invariant kernels, the $F_{-2}(\theta)$ coefficient (dominant term) can be easily derived from regular part of the kernel. Indeed, if the kernel can be written as :

$$V(\mathbf y, \mathbf x) = \mathcal V(\mathbf r)\mathbf n(\mathbf x), \quad \mathbf r = \mathbf y - \mathbf x$$

Then, the dominant term of the Laurent's expansion of $F(\rho, \theta)$ is given by :

$$F_{-2}(\theta) = \mathcal V(\mathbf A(\theta)) N^a(\bm \eta) \mathbf J(\bm \eta)$$

Where $\mathbf A(\theta) = D\tau(\bm \eta) \cdot \mathbf c(\theta)$, with $D\tau$ the Jacobian matrix of the parametric mapping $\tau$ at the singularity point $\bm \eta$ and $\mathbf J(\bm \eta)$ is the integration measure multiplied by the normal at point $\bm \eta$.

Analytical formulas for the Laplace and elastostatics hypersingular kernels are given in the [Appendix](appendix.md).

### Expansion using automatic differentiation

If the kernel verifies the translation invariance property presented previously, it is possible to compute the coefficients of the Laurent's expansion using automatic differentiation. 

In fact, we let :

$$\mathcal F(\rho, \theta) = \rho^2F(\rho, \theta)$$

So that :

$$F_{-2}(\theta) = \mathcal F(0, \theta), \quad F_{-1}(\theta) = \frac{\partial \mathcal F}{\partial \rho}(0, \theta)$$

The function $\mathcal F(\rho, \theta)$ is mathematically regular but also numerically regular at $\rho = 0$ since the singularity has been analytically removed by multiplying by $\rho^2$. The analytical formula for $\mathcal F$ is given by :

$$\mathcal F(\rho, \theta) = \frac{1}{\|\mathbf A(\rho, \theta)\|^3}\hat K(\bm x(\bm\eta), \bm x(\bm\xi(\rho, \theta)))N^a(\bm \xi(\rho, \theta)) J(\bm \xi(\rho, \theta))$$

Where $\hat K$ is the regular part of the kernel, i.e. $K(\bm y, \bm x) = \frac{1}{r^3}\hat K(\bm y, \bm x)$, and 

$$\bm A(\rho, \theta) = D\tau(\bm\eta) \cdot \mathbf c(\theta) + \frac{\rho}{2}\mathbf c(\theta)^TD^2\bm\tau(\bm\eta)\mathbf c(\theta)$$

is such that $\mathbf r = \rho \mathbf A(\rho, \theta) + \mathcal O(\rho^3)$.

Here again, we retrieve the $F_{-2}$ coefficient easily by evaluating $\mathcal F$ at $\rho = 0$. The $F_{-1}$ coefficient is obtained by computing the derivative of $\mathcal F$ with respect to $\rho$ at $\rho = 0$ using automatic differentiation.

### Expansion using a mixed approach between semi-analytical formula and Richardson extrapolation

This approach is a hybrid between the previous ones and Richardson extrapolation. The dominant term $F_{-2}(\theta)$ is computed using the analytical formula, while the $F_{-1}(\theta)$ coefficient is computed using a numerical approximation based on Richardson extrapolation:

$$F_{-2}(\theta) = \frac{1}{\|\mathbf A(0, \theta)\|^3}\hat K(\bm x(\bm\eta), \bm x(\bm\eta))N^a(\bm\eta) J(\bm\eta)$$

$$F_{-1}(\theta) = \frac{\mathrm d\mathcal F(\rho, \theta)}{\mathrm d\rho}\bigg|_{\rho = 0}=\rho^{-1}\left\{\mathcal F(\rho, \theta) - F_{-2}(\theta)\right\} + \mathcal O(\rho)$$

In a strict sense, this approximation of evaluating $\rho^{-1}\left\{\mathcal F(\rho, \theta) - F_{-2}(\theta)\right\}$ for a small value of $\rho$ is indeed a first approximation, but of first order.

We can improve the accuracy of this approximation by doing as follows:

$$(a)\quad \rho^{-1}\left\{\mathcal F(\rho, \theta) - F_{-2}(\theta)\right\} = F_{-1}(\theta) + \rho F_0(\theta) + \mathcal O(\rho^2)\\
(b)\quad (t\rho)^{-1}\left\{\mathcal F(t\rho, \theta) - F_{-2}(\theta)\right\} = F_{-1}(\theta) + t\rho F_0(\theta) + \mathcal O(\rho^2)$$

By taking the combination $(b) - t \times (a)$, we can eliminate the $O(\rho)$ term and obtain a second-order accurate approximation of $F_{-1}(\theta)$:

$$(c)\quad F_{-1}(\theta) = \frac{1}{1 - t}\rho^{-1}\left\{t^{-1}\mathcal F(t\rho, \theta) - t\mathcal F(\rho, \theta) + (t - t^{-1})F_{-2}(\theta)\right\} + E_2$$

Where $E_2 = t(1 - t)\rho^2 F_1(\theta) + \mathcal O(\rho^3)$ is the truncation error. By repeating the previous procedure, e.g. by using $(c)$ as is, then evaluating it with $\rho$ replaced by $t\rho$ and taking a suitable weighted combination of the two approximations of $F_{-1}(\theta)$ yields a new approximation whose truncation error is $O(\rho^3)$, and so on.

### Full Richardson expansion

This method consists in using the Richardson extrapolation procedure to compute both coefficients $F_{-2}(\theta)$ and $F_{-1}(\theta)$:

$$F_{-2}(\theta) = \lim_{\rho \to 0}\rho^2 F(\rho, \theta)\\
F_{-1}(\theta) = \lim_{\rho \to 0}\left\{\rho F(\rho, \theta) - \frac{F_{-2}(\theta)}{\rho}\right\}$$

These are the mathematical definitions of the coefficients, but sometimes, the function $\mathcal F(\rho, \theta) = \rho^2 F(\rho, \theta)$ cannot be obtained analytically with a stable formula. This Full Richardson extrapolation method is then performed numerically by evaluating $F(\rho, \theta)$ for a sequence of decreasing values of $\rho$ and applying the Richardson extrapolation procedure to compute the limits above ; this is the default method if we don't known any particular property of the kernel.

We have to be aware of the fact that the evaluation of $F(\rho, \theta)$ for very small values of $\rho$ can be numerically unstable if the point are too small, after a certain number of Richardson iterations.