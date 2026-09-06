# Week 3 — Bayes Decision Theory: Bayes Classifiers (1)

**Course:** KOM6110 Machine Learning and Artificial Neural Networks (Dr. Muharrem Mercimek)
**Slide deck:** `resource/ytu/KOM6110-ANN-Machine-Learning-Week-05-Bayes.pdf` (the deck is numbered Week-05; the syllabus places this in week 3)
**Textbook:** Duda, Hart and Stork, *Pattern Classification*, 2nd ed., Ch. 2 (sections 2.1 to 2.3)

**In one paragraph.** Bayesian decision theory is the reference answer to classification. If you genuinely knew every relevant probability, this theory tells you the best decision rule that exists and the lowest error rate any classifier could achieve. Everything else in the course, perceptrons, SVMs, neural networks, is an attempt to approach that bound without knowing the probabilities. So learn this week not as one method among many, but as the yardstick.

---

## 1. The setting

**Methodology:** quantify the trade-offs between classification decisions using probability and the costs that accompany those decisions.

**Assumption, and it is a strong one:** the problem is posed in probabilistic terms, and **all relevant probabilities are known.**

### The running example

An observer watches fish arrive along a conveyor belt and must predict the type of the next one. The sequence is random. Two **states of nature**:

$$\omega = \omega_1 \ (\text{sea bass}), \qquad \omega = \omega_2 \ (\text{salmon})$$

The state of nature is a random variable, because the true identity of the next fish is unpredictable.

---

## 2. Priors alone

The **prior probability** $P(\omega_j)$ is the probability that a sample comes from class $j$ *in the absence of any measurement*. For example $P(\omega_1) = 0.3$, $P(\omega_2) = 0.7$.

With no other state of nature possible,

$$P(\omega_1) + P(\omega_2) = 1$$

and if the catch is equally likely, $P(\omega_1) = P(\omega_2) = 0.5$ (uniform priors).

**Decision rule using priors only:**

$$\text{Decide } \omega_1 \text{ if } P(\omega_1) > P(\omega_2), \text{ otherwise } \omega_2$$

with error probability $P(\text{error}) = \min[P(\omega_1), P(\omega_2)]$.

This is defensible for a *single* decision. Repeating it is awkward: you would call every fish a sea bass forever, even though you know both kinds arrive. The rule uses no information about the object in front of you, which is precisely what we should fix.

---

## 3. Adding a measurement

Let $x$ be a continuous feature such as the lightness of the fish (it could equally be weight or length).

**Class-conditional probability density**, $p(x \mid \omega_j)$: the density of $x$ given that the state of nature is $\omega_j$. There is one such density per class, and the difference between $p(x \mid \omega_1)$ and $p(x \mid \omega_2)$ is exactly the difference in lightness between the sea bass and salmon populations.

Note the case convention used throughout Duda and in this course:

| Symbol | Meaning |
|---|---|
| $P(\cdot)$ capital | Probability mass, a number in $[0,1]$ |
| $p(\cdot)$ lowercase | Probability density, non-negative but not bounded by 1 |

---

## 4. Bayes formula

$$P(\omega_j \mid x) = \frac{p(x \mid \omega_j)\, P(\omega_j)}{p(x)}, \qquad \text{posterior} = \frac{\text{likelihood} \times \text{prior}}{\text{evidence}}$$

with the evidence obtained by the law of total probability:

$$p(x) = \sum_{j=1}^{2} p(x \mid \omega_j)\, P(\omega_j)$$

How to read each term:

- **Prior** $P(\omega_j)$: what you believed before looking.
- **Likelihood** $p(x \mid \omega_j)$: how well class $j$ explains the measurement you made. Other things being equal, the class with the larger likelihood is more "likely" to be the true one. Note that it is a function of $\omega_j$ for fixed $x$, which is why it is not itself a probability distribution over $x$.
- **Evidence** $p(x)$: a scale factor, identical for all classes, which only guarantees that the posteriors sum to one. **It never affects the decision**, and we will drop it in week 4.
- **Posterior** $P(\omega_j \mid x)$: what you believe after looking.

The sentence to remember: observing $x$ **converts** the prior into the posterior. The product of likelihood and prior is what determines the outcome.

---

## 5. The decision rule and its error

For an observation $x$:

$$\text{if } P(\omega_1 \mid x) > P(\omega_2 \mid x) \Rightarrow \text{decide } \omega_1, \qquad \text{if } P(\omega_1 \mid x) < P(\omega_2 \mid x) \Rightarrow \text{decide } \omega_2$$

Whenever you decide, the probability of error given that particular $x$ is the posterior of the class you rejected:

$$P(\text{error} \mid x) = \begin{cases} P(\omega_1 \mid x) & \text{if we decide } \omega_2 \\ P(\omega_2 \mid x) & \text{if we decide } \omega_1 \end{cases}$$

Choosing the larger posterior therefore leaves the smaller one as the error:

$$P(\text{error} \mid x) = \min\big[P(\omega_1 \mid x),\, P(\omega_2 \mid x)\big]$$

This is the **Bayes decision rule**, and averaging over all observations

$$P(\text{error}) = \int_{-\infty}^{\infty} P(\text{error} \mid x)\, p(x)\, dx$$

is minimized *pointwise*, for every single $x$ independently. That is the whole proof of optimality, and it is worth appreciating how little work it took. Equivalently, in terms of decision regions $\mathcal{R}_1$ and $\mathcal{R}_2$:

$$P(\text{error}) = \int_{\mathcal{R}_2} p(x \mid \omega_1) P(\omega_1)\, dx + \int_{\mathcal{R}_1} p(x \mid \omega_2) P(\omega_2)\, dx$$

The resulting minimum is the **Bayes error rate**, the lower bound no classifier can beat.

---

## 6. Generalizing in four directions

The two-class, one-feature, decide-or-else setup is too narrow. Duda generalizes it in four ways at once:

1. **More than one feature.** Use a vector $\mathbf{x}$.
2. **More than two states of nature.** Use $c$ classes.
3. **Actions other than deciding the class.** In particular this admits **rejection**: refusing to decide in ambiguous cases, which is sensible when indecision is cheaper than a mistake.
4. **A loss function more general than the probability of error.** Not all mistakes cost the same.

Formally:

| Symbol | Meaning |
|---|---|
| $\mathbf{x} \in \mathcal{R}^d$ | $d$-dimensional feature vector |
| $\{\omega_1, \dots, \omega_c\}$ | The $c$ states of nature (categories) |
| $\{\alpha_1, \dots, \alpha_a\}$ | The $a$ possible actions |
| $\lambda(\alpha_i \mid \omega_j)$ | Loss incurred for taking action $\alpha_i$ when the true state is $\omega_j$ |

Bayes formula is unchanged apart from the dimension:

$$P(\omega_j \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid \omega_j) P(\omega_j)}{p(\mathbf{x})}, \qquad p(\mathbf{x}) = \sum_{j=1}^{c} p(\mathbf{x} \mid \omega_j) P(\omega_j)$$

---

## 7. Risk

### Conditional risk

The **expected loss** of taking action $\alpha_i$ after observing $\mathbf{x}$:

$$R(\alpha_i \mid \mathbf{x}) = \sum_{j=1}^{c} \lambda(\alpha_i \mid \omega_j)\, P(\omega_j \mid \mathbf{x})$$

Each possible truth is weighted by how probable it is given the data, and by what that mistake would cost.

### Overall risk

Let $\alpha(\mathbf{x})$ be a **decision rule**: a function that names, for every observation, the action to take. Its overall risk is

$$R = \int R\big(\alpha(\mathbf{x}) \mid \mathbf{x}\big)\, p(\mathbf{x})\, d\mathbf{x}$$

where $d\mathbf{x}$ is the volume element in $\mathcal{R}^d$.

### The Bayes decision rule

Because $p(\mathbf{x}) \ge 0$ and the integrand is minimized independently at each $\mathbf{x}$:

> Compute $R(\alpha_i \mid \mathbf{x})$ for $i = 1, \dots, a$ and select the action for which it is smallest.

The resulting $R$ is the **Bayes risk**, denoted $R^*$: the best performance achievable.

---

## 8. The two-category case and the likelihood ratio

Let $\alpha_1$ mean "decide $\omega_1$", $\alpha_2$ mean "decide $\omega_2$", and abbreviate $\lambda_{ij} = \lambda(\alpha_i \mid \omega_j)$.

$$R(\alpha_1 \mid \mathbf{x}) = \lambda_{11} P(\omega_1 \mid \mathbf{x}) + \lambda_{12} P(\omega_2 \mid \mathbf{x})$$
$$R(\alpha_2 \mid \mathbf{x}) = \lambda_{21} P(\omega_1 \mid \mathbf{x}) + \lambda_{22} P(\omega_2 \mid \mathbf{x})$$

Decide $\omega_1$ when $R(\alpha_1 \mid \mathbf{x}) < R(\alpha_2 \mid \mathbf{x})$. Collecting terms:

$$(\lambda_{21} - \lambda_{11})\, P(\omega_1 \mid \mathbf{x}) > (\lambda_{12} - \lambda_{22})\, P(\omega_2 \mid \mathbf{x})$$

Now substitute the posteriors from Bayes formula. The evidence $p(\mathbf{x})$ appears on both sides and cancels:

$$(\lambda_{21} - \lambda_{11})\, p(\mathbf{x} \mid \omega_1) P(\omega_1) > (\lambda_{12} - \lambda_{22})\, p(\mathbf{x} \mid \omega_2) P(\omega_2)$$

Assuming $\lambda_{21} - \lambda_{11} > 0$ (being wrong costs more than being right, which is the only sensible case), divide to obtain the **likelihood ratio test**:

$$\boxed{\ \text{Decide } \omega_1 \text{ if } \quad \frac{p(\mathbf{x} \mid \omega_1)}{p(\mathbf{x} \mid \omega_2)} > \frac{\lambda_{12} - \lambda_{22}}{\lambda_{21} - \lambda_{11}} \cdot \frac{P(\omega_2)}{P(\omega_1)} \ }$$

otherwise decide $\omega_2$.

This form is worth dwelling on. The left side depends **only on the data**. The right side is a **constant threshold** assembled from the priors and the losses. So the whole of Bayesian decision theory for two classes reduces to: compute one number from the measurement, compare it to one number fixed in advance. Raising the cost of a false $\omega_1$ raises the threshold, demanding stronger evidence before you commit to $\omega_1$.

---

## 9. Minimum-error-rate classification

Now specialize the loss. For classification, action $\alpha_i$ means deciding $\omega_i$, so the decision is correct when $i = j$ and in error when $i \ne j$. The **zero-one (symmetrical) loss function** charges one unit for every mistake and nothing for a correct decision:

$$\lambda(\alpha_i \mid \omega_j) = \begin{cases} 0 & i = j \\ 1 & i \ne j \end{cases} \qquad i, j = 1, \dots, c$$

Substituting into the conditional risk:

$$R(\alpha_i \mid \mathbf{x}) = \sum_{j=1}^{c} \lambda(\alpha_i \mid \omega_j) P(\omega_j \mid \mathbf{x}) = \sum_{j \ne i} P(\omega_j \mid \mathbf{x}) = 1 - P(\omega_i \mid \mathbf{x})$$

The conditional risk **is** the probability of error. Minimizing it therefore means maximizing the posterior:

$$\boxed{\ \text{Decide } \omega_i \text{ if } \quad P(\omega_i \mid \mathbf{x}) > P(\omega_j \mid \mathbf{x}) \quad \text{for all } j \ne i \ }$$

This is the **maximum a posteriori (MAP)** rule. It is the special case of Bayes risk minimization under equal costs, and it is the rule you should assume whenever a problem says "minimum error rate" without specifying losses.

For two classes with zero-one loss, the general threshold collapses to $\theta = P(\omega_2)/P(\omega_1)$, and with equal priors to $\theta = 1$: pick the larger likelihood.

---

## 10. Worked example

Let $P(\omega_1) = 0.3$ and $P(\omega_2) = 0.7$. At the observed value $x$, suppose $p(x \mid \omega_1) = 0.4$ and $p(x \mid \omega_2) = 0.1$.

**Step 1. Evidence.**

$$p(x) = (0.4)(0.3) + (0.1)(0.7) = 0.12 + 0.07 = 0.19$$

**Step 2. Posteriors.**

$$P(\omega_1 \mid x) = \frac{0.12}{0.19} = 0.632, \qquad P(\omega_2 \mid x) = \frac{0.07}{0.19} = 0.368$$

They sum to 1, as the evidence guarantees.

**Step 3. Minimum error rate decision.** $0.632 > 0.368$, so decide $\omega_1$, with $P(\text{error} \mid x) = 0.368$.

Notice what happened: the prior favored $\omega_2$ better than two to one, but the likelihood ratio of 4 overturned it.

**Step 4. Now introduce asymmetric losses.** Take $\lambda_{11} = \lambda_{22} = 0$, $\lambda_{12} = 10$, $\lambda_{21} = 1$. Calling a fish $\omega_1$ when it is really $\omega_2$ is ten times as expensive as the reverse.

$$R(\alpha_1 \mid x) = 10 \times 0.368 = 3.68, \qquad R(\alpha_2 \mid x) = 1 \times 0.632 = 0.632$$

Minimum risk now says **decide $\omega_2$**, the opposite of the minimum error rate decision.

**Step 5. Cross-check with the likelihood ratio test.**

$$\frac{p(x \mid \omega_1)}{p(x \mid \omega_2)} = \frac{0.4}{0.1} = 4, \qquad \text{threshold} = \frac{10 - 0}{1 - 0} \cdot \frac{0.7}{0.3} = 23.33$$

Since $4 < 23.33$, decide $\omega_2$. The two routes agree, as they must.

The lesson is the practical one: **minimum error rate is not always what you want.** When one kind of mistake is genuinely more expensive, say a missed tumor versus a false alarm, the loss matrix moves the boundary, and it should.

---

## 11. What you should be able to do

- State Bayes formula and name all four terms, including why the evidence cannot affect a decision.
- Show that choosing the larger posterior minimizes $P(\text{error} \mid x)$ pointwise, and hence minimizes $P(\text{error})$.
- Write the conditional risk and overall risk, and define the Bayes risk.
- Derive the likelihood ratio test from the two conditional risks, showing where $p(\mathbf{x})$ cancels.
- Show that the zero-one loss reduces the conditional risk to $1 - P(\omega_i \mid \mathbf{x})$, so that minimum risk becomes MAP.
- Work a numerical example both ways (posteriors and likelihood ratio) and get the same answer.
- Explain in one sentence why the Bayes error rate is a lower bound.

---

## Related notes

- [Week 2 — Knowledge Representation and Learning Types](week-02-knowledge-representation-and-learning.md) for the similarity measures that Bayes theory now replaces with a probabilistic criterion.
- [Week 4 — Bayes Classifiers II: Discriminant Functions](week-04-bayes-classifiers-ii-discriminant-functions.md) recasts everything here as discriminant functions and evaluates them for Gaussian densities.
