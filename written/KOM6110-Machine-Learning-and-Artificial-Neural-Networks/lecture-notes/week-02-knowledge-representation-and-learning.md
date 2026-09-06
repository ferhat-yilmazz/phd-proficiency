# Week 2 — Knowledge Representation and Learning Types

**Course:** KOM6110 Machine Learning and Artificial Neural Networks (Dr. Muharrem Mercimek)
**Slide deck:** `resource/ytu/KOM6110-ANN-Machine-Learning-Week-02-Intro-Knowledge-Learning.pdf`
**Textbook:** Haykin, *Neural Networks and Learning Machines*, 3rd ed., Ch. 1 (sections on feedback, architectures, knowledge representation, learning processes and tasks)

**In one paragraph.** Week 1 gave you one neuron. This week wires neurons together and asks two questions about the result. First, structural: how are they connected, and what does a feedback loop do to the dynamics? Second, and more subtle: what does it mean for a network to *know* something? The answer is that knowledge is the numerical values of the free parameters, and there are four design rules that decide whether those values will be any good. The week closes with the taxonomy of learning and the tasks learning can be applied to.

---

## 1. Feedback

Feedback is a process in which the output of an action is returned to modify the next action. It appears in living systems, in engineered systems, and in economies. In neural networks it is the defining feature of **recurrent** architectures.

### Single-loop feedback system

For a forward path operator $\mathbf{A}$ and a feedback path operator $\mathbf{B}$:

$$y_k(n) = \frac{\mathbf{A}}{1 - \mathbf{A}\mathbf{B}} \, x_j(n)$$

The quantity $\mathbf{A}/(1 - \mathbf{A}\mathbf{B})$ is the **closed-loop operator**.

Take the simplest interesting case: $\mathbf{A}$ is a fixed weight $w$, and $\mathbf{B} = z^{-1}$ is a unit delay. Then

$$\frac{\mathbf{A}}{1 - \mathbf{A}\mathbf{B}} = \frac{w}{1 - w z^{-1}}$$

Expand as a binomial (geometric) series:

$$\frac{w}{1 - w z^{-1}} = w \sum_{l=0}^{\infty} w^l z^{-l}$$

so that

$$y_k(n) = w \sum_{l=0}^{\infty} w^l z^{-l} x_j(n) = \sum_{l=0}^{\infty} w^{\,l+1} x_j(n - l)$$

This is a first order **infinite impulse response (IIR)** filter. Read the last equation carefully: the output at time $n$ is a weighted sum of *all* past inputs, with weight $w^{l+1}$ on the sample delayed by $l$.

### Stability, decided by one number

| Condition | Behavior of $y_k(n)$ | System |
|---|---|---|
| $\lvert w \rvert < 1$ | Convergent, exponentially decaying weights | Stable, **infinite memory** with fading influence of the distant past |
| $\lvert w \rvert = 1$ | Divergence is linear | Unstable |
| $\lvert w \rvert > 1$ | Divergent, exponentially growing weights | Unstable |

The stable case is the useful one: the network remembers arbitrarily far back, but recent samples dominate. This is exactly the trade-off that makes recurrent networks powerful and hard to train.

---

## 2. Network architectures

Architecture and learning algorithm are chosen together, not independently. Three fundamental classes:

### 2.1 Single-layer feedforward

Neurons organized in layers. An input layer of **source nodes** projects directly onto an output layer of **computation nodes**, and not the other way around.

The name counts only the output layer, because the source nodes perform no computation. A network with an input layer and one output layer is therefore "single-layer", not "two-layer".

### 2.2 Multilayer feedforward

Distinguished by one or more **hidden layers**, whose nodes are hidden neurons. "Hidden" means not directly visible from either the input or the output of the network.

Why add them:

- Hidden layers let the network extract **higher-order statistics** from its input.
- The network acquires a **global perspective despite local connectivity**, because of the extra synaptic connections and the extra dimension of neural interactions.

A network is **fully connected** when every node in one layer connects to every node in the next; otherwise it is partially connected.

### 2.3 Recurrent

Distinguished by having at least one feedback loop. A recurrent network may consist of a single layer in which each neuron feeds its output back to the inputs of all the others. Self-feedback (a neuron's output returning to its own input) may or may not be present.

Feedback loops involve unit delays, which makes the network dynamic and gives it memory. The practical consequence stated in the slides is worth respecting: **networks with feedback are usually harder to train than feedforward networks.**

### Examples

| Feedforward | Recurrent / feedback |
|---|---|
| Perceptron | Hopfield network |
| Probabilistic Neural Network (PNN) | Fuzzy Cognitive Map (FCM) |
| General Regression Neural Network (GRNN) | Boltzmann Machine |

---

## 3. Knowledge

> Knowledge refers to stored information or models used by a person or machine to interpret, predict, and appropriately respond to the outside world.

Two ingredients a neural network needs: **establishing knowledge**, then **using knowledge**.

A network learns a model of its environment and must keep that model consistent with the real world. The environment is described by a set of examples, labeled or unlabeled. A set of input-output pairs, each consisting of an input signal and a desired response, is the **training data**. Labeled samples are expensive because they require a teacher, which is the practical reason unsupervised and semi-supervised methods matter.

### A concrete pipeline: handwritten letter recognition

1. **Choose the structure.** Input layer sized by the number of pixels; output layer with 26 nodes, one per letter.
2. **Train** on a labeled subset, teaching the network how to behave for the given inputs and desired values.
3. **Test** on data not seen before, and assess recognition performance against the true identities.

Success is measured by **generalization**, not by training accuracy.

### What is the network actually doing?

Instead of formulating a mathematical model of letters, the data speaks for itself. The knowledge representation of the environment is **the set of values assigned to the free parameters**, meaning the synaptic weights and biases. That is the whole answer to "where is the knowledge stored".

---

## 4. The four rules of knowledge representation

**Rule 1. Similar inputs from similar classes should produce similar representations inside the network, and should be classified as belonging to the same class.**

This requires a measure of similarity. Two standard ones, for $m$-dimensional feature vectors $\mathbf{x}_i = [x_{i1}, x_{i2}, \dots, x_{im}]^T$ defining points in $\mathcal{R}^m$:

**Euclidean distance:**

$$d(\mathbf{x}_i, \mathbf{x}_j) = \lVert \mathbf{x}_i - \mathbf{x}_j \rVert = \sqrt{\sum_{k=1}^{m} (x_{ik} - x_{jk})^2}$$

**Inner (dot) product:**

$$\langle \mathbf{x}_i, \mathbf{x}_j \rangle = \mathbf{x}_i^T \mathbf{x}_j = \sum_{k=1}^{m} x_{ik} \, x_{jk}$$

The two are linked by

$$\lVert \mathbf{x}_i - \mathbf{x}_j \rVert^2 = \lVert \mathbf{x}_i \rVert^2 + \lVert \mathbf{x}_j \rVert^2 - 2\,\mathbf{x}_i^T \mathbf{x}_j$$

so when the vectors have fixed norm (for example after normalization), **minimizing Euclidean distance is the same as maximizing the inner product.** Small distance, large inner product, more similar vectors.

*Example of a feature vector.* Measure a letter image and obtain area $= 12345$, perimeter $= 678$, mean gray level $= 73$, average blob solidity $= 0.7$. Then $\mathbf{x} = [12345, 678, 73, 0.7]^T$. Notice immediately that the scales are wildly different, which is why feature normalization is not optional if you intend to use Euclidean distance.

**Rule 2. Items to be assigned to different classes should be given widely different representations in the network.**

Rules 1 and 2 together say: compress within classes, separate between classes. That single sentence is also the objective of linear discriminant analysis and, later, of the SVM margin.

**Rule 3. If a feature is important, there should be a large number of neurons involved in representing it.**

Representation capacity should follow importance. An accurate estimate of a critical quantity needs many units, which improves accuracy and fault tolerance.

**Rule 4. Prior information and invariances should be built into the design, so the network does not have to learn them.**

Building them in produces a **specialized network** with fewer free parameters, which trains faster on less data.

### Using prior information: two mechanisms

1. **Receptive fields.** Restrict which stimuli can influence a given neuron, so each hidden neuron sees only a local region of the input.
2. **Weight sharing.** Force groups of neurons to use the same weight vector.

These two mechanisms, taken together, *are* the convolutional layer. Keep this in mind for weeks 13 and 14: the CNN is not a new idea so much as Rule 4 applied rigorously to images.

### Using invariance: three approaches

The problem: a rotated object produces a different image; a radar return varies with target motion; the same utterance may be soft or loud, slow or quick. The classifier should not care.

| Approach | Mechanism | Drawback |
|---|---|---|
| **Invariance by structure** | Build synaptic connections so that transformed versions of the same input are forced to produce the same output | Number of connections grows severely |
| **Invariance by training** | Present many transformed instances and let the network generalize | Computational demand is huge, and it must be repeated for every new class |
| **Invariance by feature space** | Extract features that are already invariant, then classify in that space | Requires designing the invariant features |

The third is usually the relief: the feature vector is invariant by construction, the number of features presented to the network can be genuinely reduced, the demands on the network are lower, and invariance is assured for all objects. **Hu's moment invariants** are the classic example. An image moment is a particular weighted average of pixel intensities chosen to have an attractive invariance property.

Closing caveat from the slides: knowledge representation is inseparable from network architecture, and **there is no well developed theory for optimizing the architecture of a neural network.** Expect empirical choices.

---

## 5. The learning process

A network learns a model of its environment by updating the values of its free parameters. After choosing the architecture, you must choose the mechanism that updates weights and biases.

$$\text{Learning} \begin{cases} \textbf{With a teacher} & \text{(supervised)} \\[2pt] \textbf{Without a teacher} & \begin{cases} \text{Unsupervised (self-organizing)} \\ \text{Reinforcement} \end{cases} \end{cases}$$

### 5.1 Learning with a teacher

- The teacher has knowledge of the environment, represented as input-output examples.
- The environment is initially unknown to the network.
- The teacher supplies the **desired response** for each input vector.
- The **error signal** is the difference between the desired output and the actual response. It is aggregated as a sum of squared errors or a mean squared error.
- Free parameters are adjusted **iteratively** to reduce that error, step by step down the error surface.
- Knowledge is thereby transferred from the teacher to the network. Once training stops, the weights are fixed and the network operates on its own.

### 5.2 Learning without a teacher (a): unsupervised learning

No external teacher oversees the process. The network adopts the **statistical regularities** of the input data and develops an internal representation that encodes input features, forming classes automatically. A task-independent measure of quality is chosen in advance, and the network is tuned to the regularities of the data with respect to it.

*Example: competitive learning.* An input layer receives the data, and a competitive layer contains neurons that compete to respond to features of the input. The neuron with the greatest total input **wins** the competition and turns on, while the others are silenced. Winner-take-all is the mechanism behind self-organizing maps and connects directly to clustering.

### 5.3 Learning without a teacher (b): reinforcement learning

The learner learns from its own experience through continued interaction with the environment, minimizing a scalar performance index (or maximizing a reward).

Nothing tells the learner which action to take. It must discover which actions are most rewarding by trying them. Two defining features:

- **Trial and error search.**
- **Delayed reward.** The consequences of an action may only become visible much later, which creates the credit assignment problem.

The learner pursues a goal despite uncertainty about the environment, and its actions are allowed to affect the future state of that environment. A chess player choosing a move is the standard illustration. This thread continues in weeks 11 and 12.

---

## 6. Learning tasks

What can be learned? Four families.

### 6.1 Pattern association

An **associative memory** is a brain-like distributed memory that learns by association. Two forms:

| Form | Description | Learning type |
|---|---|---|
| **Auto-association** | Store a set of patterns by repeated presentation; later, given a partial or noisy version, retrieve the original | Unsupervised |
| **Hetero-association** | Pair an arbitrary set of input patterns with an arbitrary set of output patterns | Supervised |

The design tension is **storage capacity** against **recall accuracy**: pushing more patterns into a fixed network eventually corrupts retrieval.

### 6.2 Pattern recognition (classification)

> Pattern recognition is the process whereby a received pattern or signal is assigned to one of a prescribed number of classes.

Humans do this almost effortlessly: recognizing a familiar face despite years of aging, or a familiar voice over a bad telephone connection. They do it through a learning process, and so does a neural network: first a training session, then presentation of a new, previously unseen pattern that the network classifies from what it extracted during training.

### 6.3 Function approximation

Given an unknown input-output mapping described by samples, find an approximation that is close in a chosen norm.

**a) System identification.** Model the input-output relation of an unknown memoryless (time invariant) MIMO system. The error between the plant output and the network output drives adjustment of the free parameters, minimizing the squared difference in a statistical sense over the entire training sample.

**b) Inverse modeling.** Construct a model that produces the *input* in response to the output. This is generally harder than identification, because **an inverse need not exist or be unique.**

Both are the direct bridge from this course to control engineering.

### 6.4 Control

Control of a plant suits neural networks well. The brain itself is an information processor whose outputs, as a whole system, are actions: it controls thousands of muscle fibers in parallel, handles nonlinearity and noise, and optimizes over a long planning horizon.

The usual recipe: from actual plant input-output measurements, first build a neural model that copies the plant, then use that model in the control loop.

> Haykin also lists **filtering** and **beamforming** as learning tasks. They are not in this deck, but they are natural for an oral exam answer if you are asked for the full list.

---

## 7. What you should be able to do

- Derive $y_k(n) = \sum_{l\ge 0} w^{l+1} x_j(n-l)$ from the closed-loop operator, and state the stability condition on $w$ with the three cases.
- Describe the three network architecture classes and explain why "single-layer" excludes the source nodes.
- Say precisely where a neural network's knowledge resides.
- State the four rules of knowledge representation and give one design consequence of each.
- Show algebraically why minimizing Euclidean distance and maximizing the inner product agree for normalized vectors.
- Compare the three routes to invariance and argue for the feature-space route.
- Draw the learning taxonomy and contrast supervised, unsupervised, and reinforcement learning by their training signal.
- Name the four learning tasks and give the auto- versus hetero-association distinction.

---

## Related notes

- [Week 1 — Introduction and Neuron Models](week-01-introduction-and-neuron-models.md) for the single neuron model this week connects together.
- [Week 3 — Bayes Decision Theory I](week-03-bayes-decision-theory-i.md) formalizes "similar inputs, same class" as an optimal statistical rule.
