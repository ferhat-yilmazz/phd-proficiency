# Week 1 — Introduction: What Is Machine Learning, ANN Representations, Biological Motivation

**Course:** KOM6110 Machine Learning and Artificial Neural Networks (Dr. Muharrem Mercimek)
**Slide deck:** `resource/ytu/KOM6110-ANN-Machine-Learning-Week-01-Introduction.pdf`
**Textbook:** Haykin, *Neural Networks and Learning Machines*, 3rd ed., Introduction and Ch. 1; Alpaydin, *Introduction to Machine Learning*, Ch. 1

> **A note on week numbering.** The lecturer's weekly plan and the slide filenames do not line up. The plan places Bayes classifiers in weeks 3 and 4, while the decks label them Week-05 and Week-06. Slide 8 of the Week-01 deck reproduces the plan exactly, so these notes follow the plan. Week 1 uses deck `Week-01`.

**In one paragraph.** Machine learning replaces a hand written model with a model fitted from data. Artificial neural networks are one family of structures for doing that, and their design borrows two ideas from biology: knowledge arrives through a learning process, and it is stored in the strengths of connections. This week builds the vocabulary (learning, generalization, synaptic weight, activation function) and gives the mathematical model of a single neuron that everything later in the course rests on.

---

## 1. What machine learning is

> Machine learning is programming computers to optimize a performance criterion using example data or past experience.

Read that definition slowly. Three parts carry the weight:

| Part | Meaning |
|---|---|
| *performance criterion* | A scalar you can measure and improve, for example squared error or misclassification rate. Without it there is nothing to optimize. |
| *optimize* | The learning algorithm is a search over free parameters. |
| *example data or past experience* | The information comes from samples, not from a designer's equations. |

The ultimate goal is a model that turns information into knowledge: something that generalizes beyond the samples it was fitted on.

### When learning is the right tool

Learning earns its cost when writing the rules by hand is impossible or impractical:

- **Human expertise does not exist.** Navigating on Mars.
- **Humans cannot articulate their expertise.** Speech recognition. You recognize a word without being able to state the rule.
- **The solution changes over time.** Routing on a computer network.
- **The solution must adapt to individual cases.** User biometrics.

The economic argument behind all four: data is cheap and abundant, explicit knowledge is expensive and scarce.

### Application areas

Retail (market basket analysis, CRM), finance (credit scoring, fraud detection), manufacturing (optimization, troubleshooting), medicine (diagnosis), telecommunications (quality of service), bioinformatics (motifs, alignment), web mining (search engines).

Classification, also called pattern recognition, is the workhorse: face recognition under pose, lighting and occlusion; character recognition across handwriting styles; speech recognition with its temporal dependency; sensor fusion combining lip images with acoustics; gesture recognition; medical diagnosis from symptoms to illnesses.

---

## 2. Where ANNs sit inside machine learning

Machine learning is the *learning* part of artificial intelligence. It is organized by what the training signal looks like:

| Type | Training signal | Typical task |
|---|---|---|
| Supervised | Input paired with a desired output (a label) | Classification, regression, function approximation |
| Unsupervised | Inputs only | Clustering, density estimation, feature discovery |
| Semi-supervised | A few labels plus many unlabeled samples | Classification with scarce labels |
| Reinforcement | A scalar reward, often delayed | Sequential decision making, game play, control |

Each of these can be carried out on many different *structures*: decision trees, kernel machines, graphical models, and artificial neural networks among them. So the relationship to hold onto is:

**ANN is a subset of machine learning, not a synonym for it.** It is one structure, inspired by functional aspects of biological neural networks, on which supervised, unsupervised, and reinforcement learning can all be run.

---

## 3. Why the brain is the reference design

The brain computes in an entirely different way from a digital computer. Its attributes:

- Highly complex, nonlinear, and massively parallel.
- Able to organize its neurons for specific computations: pattern recognition, perception, motor control.
- Silicon logic gates switch in nanoseconds; neurons operate on the millisecond scale, five to six orders of magnitude slower. The brain compensates entirely through the number of units working at once.

A perceptual recognition task such as recognizing a familiar face is completed in roughly **100 to 200 milliseconds**, which conventional computers still struggle to match on unconstrained inputs.

> **Correction to the slides.** Slide 17 gives this figure as "100-200 ns". Haykin's text says 100-200 ms. Nanoseconds would be faster than the underlying neurons, which is physically impossible. Use milliseconds.

The bat's sonar makes the same point from a different direction: from echoes a bat extracts relative velocity, target size, azimuth and elevation, and more, at a size and power budget no engineered radar approaches.

### How the brain acquires this ability

Rules and behaviors are built through experience. Much of the hardwiring of the human brain happens in the first two years after birth, and development continues for years after. The developing nervous system adapts to its surroundings, with neurons acting as the information processing units.

### Definition of a neural network

Haykin's working definition, which is what an exam answer should reproduce:

> A neural network is a machine designed to model the way in which the brain performs a particular task or function. It is implemented with electronic components or simulated in software, and it is a massively parallel distributed processor made up of simple processing units that has a natural propensity for storing experiential knowledge and making it available for use.

It resembles the brain in two respects:

1. **Knowledge is acquired by the network from its environment through a learning process.**
2. **Interneuron connection strengths, the synaptic weights, are used to store the acquired knowledge.**

Learning traditionally means modifying the synaptic weights. A network may also modify its own *topology*, motivated by the fact that neurons die and new synaptic connections grow.

---

## 4. The nervous system as a three-stage system

$$\text{Stimulus} \;\longrightarrow\; \boxed{\text{Receptors}} \;\rightleftarrows\; \boxed{\text{Neural net (brain)}} \;\rightleftarrows\; \boxed{\text{Effectors}} \;\longrightarrow\; \text{Response}$$

- **Receptors** convert stimuli from the body or the environment into electrical impulses.
- The **neural net** (the brain) continually receives information, perceives it, and makes decisions.
- **Effectors** convert the brain's electrical impulses into discernible responses.

Forward arrows carry information from stimulus to response. The reverse arrows are feedback, present at every stage. Feedback becomes the subject of week 2.

---

## 5. The biological neuron, kept to what matters

- **Dendrites** are the receptive zones; **axons** are the transmission lines.
- **Synapses** (nerve endings) are the elementary structural and functional units mediating interactions between neurons. A synaptic cleft is on the order of 20 nm.
- Transmission is chemical: a presynaptic process releases neurotransmitters (dopamine, serotonin) which diffuse across the cleft and act on the postsynaptic process. So a synapse converts an electrical signal to a chemical one and back to electrical.
- A given synapse imposes **either excitation or inhibition, not both**. This is where the artificial model departs from biology: a synaptic weight $w_{kj}$ is free to be positive or negative.
- Pyramidal cells, a common cortical neuron, can receive 10,000 or more synaptic contacts and project onto thousands of target cells.

**Structural hierarchy**, lowest to highest:

$$\text{molecules} \to \text{synapses} \to \text{neural microcircuits} \to \text{dendritic trees} \to \text{neurons} \to \text{local circuits} \to \text{interregional circuits} \to \text{CNS}$$

**Two mechanisms of adaptation:** creation of new synaptic connections, and modification of existing synapses.

---

## 6. Model of a neuron

A neuron is an information processing unit with **three basic elements**:

1. **A set of synapses**, each with its own weight. A signal $x_j$ at the input of synapse $j$ connected to neuron $k$ is multiplied by $w_{kj}$. Index order matters: the first subscript is the receiving neuron, the second is the source.
2. **An adder** (linear combiner) summing the weighted inputs.
3. **An activation function** limiting the amplitude of the output, typically to $[0,1]$ or $[-1,1]$.

An externally applied **bias** $b_k$ shifts the input to the activation function.

### First representation

$$u_k = \sum_{j=1}^{m} w_{kj} x_j, \qquad y_k = \varphi(u_k + b_k)$$

where $u_k$ is the linear combiner output and $v_k = u_k + b_k$ is the **induced local field**, also called the activation potential.

### Second representation: absorbing the bias

Add one more synapse fixed at $x_0 = +1$ with weight $w_{k0} = b_k$. Then

$$v_k = \sum_{j=0}^{m} w_{kj} x_j, \qquad y_k = \varphi(v_k)$$

Nothing changes numerically, but the bias is now just another weight. Every learning rule in this course therefore trains the bias exactly the way it trains a weight, with no special case. This trick reappears in the perceptron, in linear discriminant functions, and in SVMs. It is worth being fluent in it.

In vector form, with $\mathbf{w}_k = [w_{k0}, w_{k1}, \dots, w_{km}]^T$ and $\mathbf{x} = [1, x_1, \dots, x_m]^T$:

$$y_k = \varphi(\mathbf{w}_k^T \mathbf{x})$$

---

## 7. Activation functions

**1. Threshold (Heaviside, McCulloch-Pitts).** All or none.

$$\varphi(v) = \begin{cases} 1 & v \ge 0 \\ 0 & v < 0 \end{cases}$$

**2. Sigmoid (logistic).** The most common in practice, with slope parameter $a$:

$$\varphi(v) = \frac{1}{1 + e^{-av}} \in (0, 1)$$

**3. Signum.** The odd symmetric counterpart of the threshold, with range $\{-1, 0, +1\}$.

**4. Hyperbolic tangent.** The odd symmetric counterpart of the sigmoid:

$$\varphi(v) = \tanh(v) \in (-1, 1)$$

**The point that matters for the rest of the course:** the sigmoid and $\tanh$ take a continuous range of values and are **differentiable**; the threshold and signum are not. Differentiability is what allows gradient based training, so it is a precondition for the delta rule and backpropagation in weeks 7 and 9. Note also the convenient derivative $\varphi'(v) = \varphi(v)\,[1 - \varphi(v)]$ for the logistic sigmoid with $a = 1$.

---

## 8. Neural networks as directed graphs

Signal flow graphs, developed by Mason for linear networks, give a compact picture of a network. Nonlinearity limits how far the classical rules carry, but the notation is still a neat way to show structure.

A neural network is a directed graph characterized by four properties:

1. Each neuron is a set of **linear synaptic links**, an externally applied **bias**, and a possibly nonlinear **activation link**. The bias is a synaptic link whose input is fixed at $+1$.
2. The synaptic links weight their respective input signals.
3. The weighted sum of the input signals defines the **induced local field** of the neuron.
4. The activation link squashes the induced local field to produce the output.

A graph that shows only the layout, not every internal detail, is an **architectural graph**. The simplest case is a single neuron with $m$ source nodes plus one node fixed at $+1$ for the bias.

---

## 9. Properties and capabilities of neural networks

Nine items, worth memorizing as a list because they are natural exam material.

1. **Nonlinearity.** A neuron can be linear or nonlinear, and the nonlinearity is *distributed* through the network rather than concentrated in one place. This matters when the physical signal itself is nonlinear.
2. **Input-output mapping.** In supervised learning, labeled examples are presented, and the free parameters are modified to reduce the difference between the desired and actual response under a statistical criterion. Training repeats until the weights reach a steady state. The network builds the input-output map from data rather than from assumptions.
3. **Adaptivity.** Weights can be retrained for minor environmental changes, and in a nonstationary environment they can be adapted in real time. (The design tension: adapt too fast and the network tracks noise.)
4. **Evidential response.** A classifier can report not just the selected class but the confidence in it, which supports rejecting ambiguous patterns.
5. **Contextual information.** Knowledge lives in the structure and activation state of the whole network. Every neuron is potentially affected by every other, so context is handled naturally.
6. **Fault tolerance.** Performance degrades gracefully rather than catastrophically when units or connections are damaged, because the representation is distributed.
7. **VLSI implementability.** Massive parallelism maps well onto very large scale integrated hardware.
8. **Uniformity of analysis and design.** The same notation, the same neurons, and the same learning theories are shared across applications, and modular networks can be assembled seamlessly.
9. **Neurobiological analogy.** Biology is a source of design ideas, and engineered systems return the favor as models. The retina, which converts an optical image to a neural image through a specific synaptic organization, has been imitated in silicon chips.

Two of these deserve emphasis because they are why neural networks are used at all: **computing power** from the massively parallel structure and the ability to learn, and **generalization**, meaning production of reasonable outputs for inputs never encountered during training. Together they yield good approximations to complex problems.

A caution the lecturer states directly: a neural network does not solve a problem on its own. It needs a consistent system engineering approach around it, from feature choice to validation.

---

## 10. What you should be able to do

- State the definition of machine learning and the four situations in which learning is the appropriate tool.
- Explain precisely why ANNs are a subset of machine learning, not an alternative to it.
- Give Haykin's two-point definition of a neural network (knowledge acquired by learning, knowledge stored in synaptic weights).
- Draw the three-stage nervous system diagram, including the feedback paths.
- Write the neuron model in both representations, and show that absorbing the bias as $w_{k0}$ with $x_0 = +1$ leaves the output unchanged.
- Sketch the four activation functions and say which are differentiable and why that matters.
- List the four properties defining a neural network as a directed graph.
- Recite and briefly explain the nine capabilities.

---

## Related notes

- [Week 2 — Knowledge Representation and Learning Types](week-02-knowledge-representation-and-learning.md) extends this to feedback, network architectures, and the learning process.
