# Week 4 — Bayes Classifiers (2): Discriminant Functions

**Course:** KOM6110 Machine Learning and Artificial Neural Networks (Dr. Muharrem Mercimek)
**Slide deck:** `resource/ytu/KOM6110-ANN-Machine-Learning-Week-06-Bayes-II.pdf` (the deck is numbered Week-06; the syllabus places this in week 4)
**Textbook:** Duda, Hart and Stork, *Pattern Classification*, 2nd ed., Ch. 2 (sections 2.4 to 2.6)

**In one paragraph.** Week 3 produced an optimal rule stated in probabilities. This week rewrites it in a form a machine can compute: a set of $c$ scoring functions, one per class, with the largest score winning. That reframing is what makes the connection to neural networks visible. Then we evaluate those functions for Gaussian densities and find something important: depending only on the covariance structure, the optimal Bayes classifier becomes a linear machine, a Mahalanobis nearest-mean rule, or a quadratic machine. The linear cases are literally single-layer perceptrons, which is why this material sits directly before perceptrons in the course.

---

## 1. Discriminant functions

Represent a classifier by a set of **discriminant functions** $g_i(\mathbf{x})$, $i = 1, \dots, c$, with the rule

$$\text{assign } \mathbf{x} \text{ to } \omega_i \quad \text{if} \quad g_i(\mathbf{x}) > g_j(\mathbf{x}) \ \ \text{for all } j \ne i$$

The classifier is then a machine that computes $c$ scores and selects the largest. Picture it as a network: inputs $x_1, \dots, x_d$ fan out to $c$ units computing $g_1, \dots, g_c$, followed by a maximum selector. That picture is the whole reason this topic belongs in a neural networks course.

### Bayes classifiers written this way

Two natural choices:

| Goal | Discriminant |
|---|---|
| Minimum risk | $g_i(\mathbf{x}) = -R(\alpha_i \mid \mathbf{x})$, so maximum discriminant is minimum risk |
| Minimum error rate | $g_i(\mathbf{x}) = P(\omega_i \mid \mathbf{x})$, so maximum discriminant is maximum posterior |

### Discriminants are not unique

The rule only compares scores, so **any strictly monotonically increasing function $f$ applied to every $g_i$ leaves the classifier unchanged.** Use this freedom to simplify.

Starting from $g_i(\mathbf{x}) = P(\omega_i \mid \mathbf{x}) = p(\mathbf{x}\mid\omega_i)P(\omega_i) / p(\mathbf{x})$:

- Drop the evidence $p(\mathbf{x})$, which is common to all $i$:

$$g_i(\mathbf{x}) = p(\mathbf{x} \mid \omega_i)\, P(\omega_i)$$

- Take the natural logarithm, which is monotone increasing and turns products into sums:

$$\boxed{\ g_i(\mathbf{x}) = \ln p(\mathbf{x} \mid \omega_i) + \ln P(\omega_i)\ }$$

This last form is the one used for the rest of the week. Logarithms are the right move because the Gaussian density is an exponential, and taking the log of it leaves a clean quadratic form.

---

## 2. Decision regions and boundaries

The discriminant functions partition the feature space into $c$ **decision regions** $\mathcal{R}_1, \dots, \mathcal{R}_c$:

$$\mathbf{x} \in \mathcal{R}_i \quad \Longleftrightarrow \quad g_i(\mathbf{x}) > g_j(\mathbf{x}) \ \ \forall j \ne i$$

Every point in $\mathcal{R}_i$ is labeled $\omega_i$. The regions are separated by **decision boundaries** (decision surfaces), defined by ties:

$$g_i(\mathbf{x}) = g_j(\mathbf{x})$$

Two things to note now, because both surprise people later. A decision region **need not be connected**, and a boundary is a surface of dimension $d-1$ in a $d$-dimensional feature space, so in 2D it is a curve and in 3D a surface.

---

## 3. The two-category case: the dichotomizer

With only two classes it is more common to use a **single** discriminant function, called a dichotomizer:

$$g(\mathbf{x}) = g_1(\mathbf{x}) - g_2(\mathbf{x})$$

Decision rule: **choose $\omega_1$ if $g(\mathbf{x}) > 0$, otherwise $\omega_2$.**

Written out with the log form:

$$g(\mathbf{x}) = \ln p(\mathbf{x} \mid \omega_1) + \ln P(\omega_1) - \ln p(\mathbf{x} \mid \omega_2) - \ln P(\omega_2)$$

$$g(\mathbf{x}) = \ln \frac{p(\mathbf{x} \mid \omega_1)}{p(\mathbf{x} \mid \omega_2)} + \ln \frac{P(\omega_1)}{P(\omega_2)}$$

This is the **log-likelihood ratio test** from week 3, now with the threshold moved to the left side and compared against zero. It is the same rule in different clothing, and recognizing that is worth a mark on an exam.

---

## 4. The normal density

### Why Gaussians

They describe many natural populations well, and specifically they are the right model when patterns are **random variations of an ideal prototype**, where the prototype is the mean vector and the variation is noise. Height and weight in a population are everyday examples.

### Univariate

$$p(x) = \frac{1}{\sqrt{2\pi}\, \sigma} \exp\left[-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2\right]$$

$$\mu = \mathcal{E}[x] = \int_{-\infty}^{\infty} x\, p(x)\, dx, \qquad \sigma^2 = \mathcal{E}\big[(x - \mu)^2\big] = \int_{-\infty}^{\infty} (x-\mu)^2 p(x)\, dx$$

Two parameters fully specify it: $p(x) = N(\mu, \sigma^2)$. When it is a class-conditional density we write $p(x \mid \omega_1) = N(\mu_1, \sigma_1^2)$.

### Multivariate

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} \lvert \boldsymbol{\Sigma} \rvert^{1/2}} \exp\left[-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right]$$

with $\mathbf{x}$ a $d$-component random vector, $\boldsymbol{\mu} = \mathcal{E}[\mathbf{x}]$ the mean vector, and $\boldsymbol{\Sigma} = \mathcal{E}[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^T]$ the $d \times d$ covariance matrix. Compactly $p(\mathbf{x}) = N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.

The quantity in the exponent,

$$r^2 = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$$

is the squared **Mahalanobis distance** from $\mathbf{x}$ to $\boldsymbol{\mu}$. It is Euclidean distance rescaled by the spread of the data in each direction, so a deviation along a direction of large variance counts for less. Contours of constant density are hyperellipsoids of constant Mahalanobis distance.

---

## 5. Worked example: two univariate Gaussians

This is the example from the slides, carried through to numbers, because the arithmetic is the part that goes wrong under exam pressure.

**Given.** $p(x \mid \omega_1) = N(0, 3^2)$ and $p(x \mid \omega_2) = N(2, 1^2)$, with $P(\omega_1) = P(\omega_2)$ and zero-one loss.

**Step 1. Build the dichotomizer.** Equal priors kill the prior term:

$$g(x) = \ln \frac{p(x \mid \omega_1)}{p(x \mid \omega_2)} + \underbrace{\ln \frac{P(\omega_1)}{P(\omega_2)}}_{= \,0}$$

**Step 2. Substitute the densities.**

$$g(x) = \left[-\ln(\sqrt{2\pi}\cdot 3) - \frac{x^2}{2 \cdot 9}\right] - \left[-\ln(\sqrt{2\pi}\cdot 1) - \frac{(x-2)^2}{2 \cdot 1}\right] = -\ln 3 - \frac{x^2}{18} + \frac{(x-2)^2}{2}$$

The $\sqrt{2\pi}$ cancels; only the ratio $\sigma_2/\sigma_1$ survives as $-\ln 3$.

**Step 3. Find the boundary, $g(x) = 0$.** Multiply through by 18:

$$-18\ln 3 - x^2 + 9(x-2)^2 = 0 \ \Longrightarrow\ 8x^2 - 36x + 36 - 18\ln 3 = 0$$

With $18 \ln 3 = 19.775$:

$$8x^2 - 36x + 16.225 = 0 \ \Longrightarrow\ x = \frac{36 \pm \sqrt{1296 - 519.2}}{16} = \frac{36 \pm 27.871}{16}$$

$$x_a = 0.508, \qquad x_b = 3.992$$

**Step 4. Assign the regions.** Test one point in each interval:

| Test point | $g(x)$ | Decision |
|---|---|---|
| $x = 0$ | $-1.099 - 0 + 2 = +0.901$ | $\omega_1$ |
| $x = 2$ | $-1.099 - 0.222 + 0 = -1.321$ | $\omega_2$ |
| $x = 10$ | $-1.099 - 5.556 + 32 = +25.35$ | $\omega_1$ |

$$\mathcal{R}_1 = (-\infty,\ 0.508) \cup (3.992,\ \infty), \qquad \mathcal{R}_2 = (0.508,\ 3.992)$$

**Step 5. Read the result.** There are **two** boundary points, and $\mathcal{R}_1$ is **not connected**. This is the promised surprise from section 2, and the reason is that the variances differ. Class 2 is narrow and tall, so it wins near its own mean, while the broad class 1 wins in both tails. A single threshold could never express this. Whenever $\sigma_1 \ne \sigma_2$, expect a quadratic boundary equation and therefore up to two roots.

> **Check yourself with the 2023 final exam.** The same question appears there with $p(x\mid\omega_1) = N(0,1)$ and $p(x\mid\omega_2) = N(1,4)$. Following the identical five steps you should get $3x^2 + 2x - 6.545 = 0$, hence boundaries at $x = -1.848$ and $x = 1.181$, with $\mathcal{R}_1$ the interval **between** them this time, because now class 1 is the narrow one. The part (b) answer: unequal priors add the constant $\ln[P(\omega_1)/P(\omega_2)]$ to $g(x)$, which shifts both roots and enlarges the region of the more probable class without changing the quadratic's shape.

---

## 6. Discriminant functions for the normal density

Substitute $p(\mathbf{x} \mid \omega_i) = N(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$ into $g_i(\mathbf{x}) = \ln p(\mathbf{x}\mid\omega_i) + \ln P(\omega_i)$:

$$g_i(\mathbf{x}) = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x}-\boldsymbol{\mu}_i) - \frac{d}{2}\ln 2\pi - \frac{1}{2}\ln \lvert \boldsymbol{\Sigma}_i \rvert + \ln P(\omega_i)$$

**The whole strategy for the three cases that follow is one idea:** delete every term that does not depend on $i$, because a constant added to all discriminants cannot change which one is largest. What remains is the smallest expression that still gives the correct decision, and its algebraic form (linear or quadratic) tells you the shape of the decision surface.

Note that $\tfrac{d}{2}\ln 2\pi$ is independent of $i$ in all three cases and is always dropped.

---

### Case 1: $\boldsymbol{\Sigma}_i = \sigma^2 \mathbf{I}$

The simplest case. Features are statistically independent and each has the same variance $\sigma^2$, so the samples of every class fall in equal-size hyperspherical clusters.

Here $\lvert \boldsymbol{\Sigma}_i \rvert = \sigma^{2d}$ and $\boldsymbol{\Sigma}_i^{-1} = \tfrac{1}{\sigma^2}\mathbf{I}$, both independent of $i$, so the determinant term drops too:

$$g_i(\mathbf{x}) = -\frac{\lVert \mathbf{x} - \boldsymbol{\mu}_i \rVert^2}{2\sigma^2} + \ln P(\omega_i)$$

**If the priors are also equal**, this is a **minimum distance classifier**: assign $\mathbf{x}$ to the class whose mean is nearest in Euclidean distance. This is template matching, and it is reassuring that the optimal statistical rule reduces to something so intuitive.

Expanding the norm:

$$\lVert \mathbf{x} - \boldsymbol{\mu}_i \rVert^2 = \mathbf{x}^T\mathbf{x} - 2\boldsymbol{\mu}_i^T \mathbf{x} + \boldsymbol{\mu}_i^T \boldsymbol{\mu}_i$$

The quadratic term $\mathbf{x}^T\mathbf{x}$ is the same for every $i$, so it goes as well. What is left is **linear in $\mathbf{x}$**:

$$g_i(\mathbf{x}) = \mathbf{w}_i^T \mathbf{x} + w_{i0}, \qquad \mathbf{w}_i = \frac{\boldsymbol{\mu}_i}{\sigma^2}, \qquad w_{i0} = -\frac{\boldsymbol{\mu}_i^T \boldsymbol{\mu}_i}{2\sigma^2} + \ln P(\omega_i)$$

$w_{i0}$ is the **threshold** or bias of the $i$-th discriminant. Compare this with the neuron model of week 1: it is exactly $\varphi(\mathbf{w}^T\mathbf{x} + b)$ without the squashing. A classifier of this form is called a **linear machine**.

**Decision surface.** Setting $g_i(\mathbf{x}) = g_j(\mathbf{x})$ gives a hyperplane that can be written

$$\mathbf{w}^T(\mathbf{x} - \mathbf{x}_0) = 0$$

$$\mathbf{w} = \boldsymbol{\mu}_i - \boldsymbol{\mu}_j, \qquad \mathbf{x}_0 = \frac{1}{2}(\boldsymbol{\mu}_i + \boldsymbol{\mu}_j) - \frac{\sigma^2}{\lVert \boldsymbol{\mu}_i - \boldsymbol{\mu}_j \rVert^2} \ln\frac{P(\omega_i)}{P(\omega_j)} (\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)$$

Properties:

- It **passes through $\mathbf{x}_0$**.
- It is **orthogonal to the line linking the means**, because $\mathbf{w} \parallel (\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)$.
- If $P(\omega_i) = P(\omega_j)$, the log term vanishes and $\mathbf{x}_0$ is the **midpoint**: the perpendicular bisector.
- If $P(\omega_i) \ne P(\omega_j)$, then $\mathbf{x}_0$ **shifts away from the mean of the more probable class**, enlarging that class's region. This is the prior doing its job.
- If $\sigma^2$ is very small relative to the separation of the means, the shift is negligible and the boundary is insensitive to the priors. Sharp, well separated classes leave little for the prior to decide.

---

### Case 2: $\boldsymbol{\Sigma}_i = \boldsymbol{\Sigma}$

Covariance matrices are identical across classes but otherwise arbitrary. Clusters are hyperellipsoids of the **same shape, size, and orientation**, differing only in location.

Both $\tfrac{d}{2}\ln 2\pi$ and $\tfrac{1}{2}\ln\lvert\boldsymbol{\Sigma}\rvert$ are now independent of $i$:

$$g_i(\mathbf{x}) = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}_i) + \ln P(\omega_i)$$

**If the priors are equal**, this is a **minimum Mahalanobis distance classifier**: compute the Mahalanobis distance from $\mathbf{x}$ to each of the $c$ mean vectors, and assign it to the nearest one. Case 1 is the special case where the Mahalanobis distance reduces to the Euclidean one.

Expanding:

$$(\mathbf{x}-\boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}_i) = \mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x} - 2\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\mathbf{x} + \boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i$$

The quadratic term $\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x}$ is independent of $i$ and drops. Again **linear**:

$$g_i(\mathbf{x}) = \mathbf{w}_i^T\mathbf{x} + w_{i0}, \qquad \mathbf{w}_i = \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i, \qquad w_{i0} = -\frac{1}{2}\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i + \ln P(\omega_i)$$

**Decision surface.** Still a hyperplane $\mathbf{w}^T(\mathbf{x}-\mathbf{x}_0) = 0$, but now

$$\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_i - \boldsymbol{\mu}_j), \qquad \mathbf{x}_0 = \frac{1}{2}(\boldsymbol{\mu}_i + \boldsymbol{\mu}_j) - \frac{\ln\big[P(\omega_i)/P(\omega_j)\big]}{(\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)}(\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)$$

Properties:

- It passes through $\mathbf{x}_0$.
- It is **generally not orthogonal** to the line linking the means, because $\mathbf{w} = \boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)$ has been rotated by $\boldsymbol{\Sigma}^{-1}$. This is the one difference from Case 1 that examiners like, and it makes geometric sense: the boundary tilts to respect the directions in which the data actually spreads.
- With unequal priors, $\mathbf{x}_0$ again shifts away from the more probable mean, though now along $(\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)$ while the hyperplane's normal points elsewhere.

---

### Case 3: $\boldsymbol{\Sigma}_i$ arbitrary

Covariance matrices differ between classes. Only $\tfrac{d}{2}\ln 2\pi$ can be dropped. The $\mathbf{x}^T\boldsymbol{\Sigma}_i^{-1}\mathbf{x}$ term now **depends on $i$ and must be kept**, so the discriminant stays quadratic:

$$g_i(\mathbf{x}) = \mathbf{x}^T \mathbf{W}_i \mathbf{x} + \mathbf{w}_i^T \mathbf{x} + w_{i0}$$

$$\mathbf{W}_i = -\frac{1}{2}\boldsymbol{\Sigma}_i^{-1}, \qquad \mathbf{w}_i = \boldsymbol{\Sigma}_i^{-1}\boldsymbol{\mu}_i, \qquad w_{i0} = -\frac{1}{2}\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}_i^{-1}\boldsymbol{\mu}_i - \frac{1}{2}\ln\lvert \boldsymbol{\Sigma}_i \rvert + \ln P(\omega_i)$$

**Decision surfaces.** Because of the quadratic term the surfaces are no longer linear. They are **hyperquadrics**, and they can take any of the general forms: hyperplanes, pairs of hyperplanes, hyperspheres, hyperellipsoids, hyperparaboloids, and hyperhyperboloids of various types.

Consequently **decision regions vary in shape and need not be connected.** The one-dimensional example in section 5 is precisely this case with $d = 1$, which is why it produced two boundary points and a split $\mathcal{R}_1$. Also, the boundary between two classes will generally not pass through the midpoint of the two means, even with equal priors.

---

### Summary of the three cases

| | Case 1: $\sigma^2\mathbf{I}$ | Case 2: $\boldsymbol{\Sigma}$ common | Case 3: $\boldsymbol{\Sigma}_i$ arbitrary |
|---|---|---|---|
| Cluster shape | Equal hyperspheres | Identical hyperellipsoids | Arbitrary hyperellipsoids |
| $g_i(\mathbf{x})$ | Linear | Linear | Quadratic |
| Rule with equal priors | Minimum Euclidean distance to mean | Minimum Mahalanobis distance to mean | Full quadratic score |
| Boundary | Hyperplane, orthogonal to $\boldsymbol{\mu}_i - \boldsymbol{\mu}_j$ | Hyperplane, generally not orthogonal | Hyperquadric |
| $\mathbf{w}_i$ | $\boldsymbol{\mu}_i / \sigma^2$ | $\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i$ | $\boldsymbol{\Sigma}_i^{-1}\boldsymbol{\mu}_i$ |
| $w_{i0}$ | $-\dfrac{\boldsymbol{\mu}_i^T\boldsymbol{\mu}_i}{2\sigma^2} + \ln P(\omega_i)$ | $-\dfrac{1}{2}\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i + \ln P(\omega_i)$ | $-\dfrac{1}{2}\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}_i^{-1}\boldsymbol{\mu}_i - \dfrac{1}{2}\ln\lvert\boldsymbol{\Sigma}_i\rvert + \ln P(\omega_i)$ |

The line to take away: **cases 1 and 2 give linear machines.** When the Gaussian assumption with shared covariance holds, the optimal Bayes classifier *is* a single-layer linear unit. Weeks 7 and 10 will train that same linear form directly from data, without ever estimating a density. Understanding that these are two routes to the same object is one of the central ideas of the course.

---

## 7. Corrections to the slide deck

Three transcription slips in the Week-06 deck. Use the corrected forms.

| Slide | As printed | Correct |
|---|---|---|
| 16 (Case 1, $\mathbf{x}_0$) | $\dfrac{\sigma^2}{\lVert \mathbf{x} - \boldsymbol{\mu}_i \rVert^2}$ | $\dfrac{\sigma^2}{\lVert \boldsymbol{\mu}_i - \boldsymbol{\mu}_j \rVert^2}$ |
| 23 (Case 2, $w_{i0}$) | $\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i + \ln P(\omega_i)$ | $-\tfrac{1}{2}\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i + \ln P(\omega_i)$ |
| 24 (Case 2, $\mathbf{x}_0$) | $\dfrac{1}{(\mathbf{x} - \boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}_i)}$ | $\dfrac{1}{(\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)}$ |

The reasoning in each case is the same: $\mathbf{x}_0$ is a fixed point on the boundary, so its definition cannot contain the variable $\mathbf{x}$. And the missing $-\tfrac{1}{2}$ follows directly from expanding the quadratic form, as shown in Case 2 above.

---

## 8. What you should be able to do

- Define a discriminant function classifier and explain why any monotone increasing transformation of all $g_i$ leaves it unchanged.
- Reduce $P(\omega_i \mid \mathbf{x})$ to $\ln p(\mathbf{x}\mid\omega_i) + \ln P(\omega_i)$, justifying each step.
- Write the dichotomizer and show it is the log-likelihood ratio test.
- Write the multivariate normal density and identify the Mahalanobis distance in its exponent.
- Given two univariate Gaussians and priors, construct $g(x)$, solve $g(x)=0$, and label the regions correctly, including the disconnected case.
- Derive $\mathbf{w}_i$ and $w_{i0}$ for Cases 1 and 2, stating at each step which terms are dropped and why.
- State the geometry of the boundary in each case, especially that Case 1 is orthogonal to the line joining the means while Case 2 generally is not, and which way $\mathbf{x}_0$ moves when the priors are unequal.
- Explain why Case 3 gives hyperquadrics and possibly disconnected regions.
- Connect Cases 1 and 2 to the linear neuron model of week 1.

---

## Related notes

- [Week 3 — Bayes Decision Theory I](week-03-bayes-decision-theory-i.md) for the risk and likelihood ratio material this note reformulates.
- [Week 1 — Introduction and Neuron Models](week-01-introduction-and-neuron-models.md) for the linear unit $\mathbf{w}^T\mathbf{x} + b$ that Cases 1 and 2 reproduce.
