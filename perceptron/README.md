# Rosenblatt's Perceptron Algorithm

An interactive visualization of Rosenblatt's perceptron learning algorithm.

[▶ Launch Interactive Demo](https://cadyehoffmann.github.io/deep-learning-demos/perceptron/)

This demo shows how a perceptron learns a linear decision boundary by repeatedly identifying misclassified training examples and updating its weight vector.

Based on the ideas introduced in [Rumelhart 1986](https://papers.baulab.info/papers/Rumelhart-1986.pdf).

## The Perceptron

For an input

x = (x₁, x₂)

the perceptron calculates the score

wᵀx + b

where:

- **w = (w₁, w₂)** is the weight vector
- **x = (x₁, x₂)** is the input
- **b** is the bias

The prediction is determined by the sign of the score:

- **wᵀx + b > 0** → predict +1
- **wᵀx + b < 0** → predict -1

A training example is misclassified when

y(wᵀx + b) ≤ 0

where **y ∈ {-1, +1}** is the example's true label.

## Learning From a Mistake

When the perceptron encounters a misclassified example, it updates its parameters using

w_new = w + yx

and

b_new = b + y

The term **yx** is the correction contributed by the misclassified example.

If **y = +1**, the correction points in the same direction as x.

If **y = -1**, multiplying by -1 reverses the direction of x.

The perceptron processes one mistake at a time and repeatedly updates its parameters until no training examples are misclassified.

## The Visualization

The demo shows the same perceptron from two complementary perspectives.

### Data Space

The left panel displays the training examples in (x₁, x₂) space.

- Filled blue points have label **y = +1**
- Hollow points have label **y = -1**
- The orange line is the perceptron's current decision boundary
- The orange **w** arrow is the current weight vector and is perpendicular to the decision boundary
- The dashed teal line shows a known separating boundary for the generated dataset
- Red rings identify examples currently misclassified by the perceptron

During an update, the red **yx** vector shows the correction contributed by the selected mistake and the purple **w_new** vector previews the new weight direction.

### Weight Space

The right panel displays the model in terms of its weights rather than its training examples.

The axes represent **w₁** and **w₂**. The orange **w** therefore shows the perceptron's current weight direction directly in weight space.

The teal **w\*** is a reference weight direction associated with the known separator in Data Space. It is not known to the perceptron and is not used to calculate its updates.

When an update is proposed, the purple **w_new** shows the new weight direction produced by

w_new = w + yx

Applying the update moves **w** in Weight Space while simultaneously changing the decision boundary shown in Data Space.

## Interaction

Use the main button to step through one perceptron update:

**Find Mistakes → Show Update → Apply Update**

1. **Find Mistakes** identifies all examples the current classifier gets wrong and selects one for the next update.
2. **Show Update** visualizes the correction and proposed new weight vector.
3. **Apply Update** updates the model and redraws the classifier.

Repeat the process to watch the perceptron learn from its mistakes.

## Purpose

This project was built as an interactive learning tool for understanding the geometry behind the perceptron algorithm, particularly the relationship between:

- the weight vector
- the decision boundary
- misclassified examples
- the perceptron update rule
- data space and weight space

It is part of a collection of interactive visualizations for understanding deep learning algorithms and papers.
