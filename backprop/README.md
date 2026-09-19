# Backpropagation Explorer

An interactive visualization of the forward pass, backpropagation, and gradient descent in a small neural network.

[▶ Launch Interactive Demo](https://cadyehoffmann.github.io/deep-learning-demos/backprop/)

This demo shows how values move forward through a neural network to produce a prediction and loss, then traces how gradients move backward through the same computation graph using the chain rule.

Based on ideas introduced in [Baydin 2018](https://papers.baulab.info/papers/also/Baydin-2018.pdf).

## The Network

The network contains:

- **2 inputs:** x₁ and x₂
- **2 hidden neurons:** h₁ and h₂
- **1 output:** ŷ
- **Sigmoid activations**
- **Squared-error loss**

For the first hidden neuron,

z₁ = w₁x₁ + w₂x₂ + b₁

and

h₁ = σ(z₁)

The second hidden neuron is computed similarly:

z₂ = w₃x₁ + w₄x₂ + b₂

and

h₂ = σ(z₂)

The output neuron combines the hidden activations:

z₃ = w₅h₁ + w₆h₂ + b₃

and produces the prediction

ŷ = σ(z₃)

The loss is

L = ½(ŷ - y)²

where **y** is the target value.

## Forward Propagation

The forward pass evaluates the network from inputs to loss:

**x₁, x₂ → hidden layer → output → loss**

The demo walks through each computation individually, showing both the numerical values in the network and a detailed calculation for the current operation.

This makes it possible to see the distinction between the weighted sums **z₁, z₂, z₃** and the activated neuron values **h₁, h₂, ŷ**.

## Backpropagation

Backpropagation moves through the computation graph in the opposite direction.

At each operation, the chain rule can be understood as:

**upstream gradient × local derivative = whole gradient**

The **upstream gradient** measures how the loss responds to the value farther downstream.

The **local derivative** measures how the current operation responds to its own input.

Multiplying them produces the gradient that continues backward through the network.

The demo exposes these calculations one operation at a time so that each numerical gradient can be traced back to the derivatives that produced it.

## The "Penny" Question

The Penny Question gives an intuitive interpretation of a gradient.

For a weight w, the derivative

∂L/∂w

describes how sensitive the loss is to a small change in that weight.

If the weight changes by 0.01, the resulting change in loss can be approximated by

ΔL ≈ (∂L/∂w)Δw

This connects the derivative calculated during backpropagation to the question:

**If this weight changed by one penny, approximately how much would the loss change?**

## Gradient Descent

Once backpropagation has calculated the gradients, the network updates its parameters using gradient descent:

w_new = w_old - η(∂L/∂w)

where **η** is the learning rate.

The gradient tells us how changing a parameter affects the loss. Gradient descent moves the parameter in the opposite direction of that gradient in order to reduce the loss.

## Watching the Network Learn

The demo can continue training the network for multiple iterations.

The training history tracks:

- prediction ŷ
- loss
- weights w₁ through w₆
- biases b₁ through b₃

This makes it possible to watch the prediction move toward the target, the loss decrease, and the individual parameters change as gradient descent repeatedly updates the network.

## Interaction

The demo allows you to:

- Step through the forward pass one operation at a time
- Step backward through the gradient calculations
- Inspect upstream gradients and local derivatives
- Use the Penny Question to interpret individual weight gradients
- Apply a gradient-descent update
- Train the network for 1, 25, or 100 iterations
- Change the inputs, target, learning rate, and biases

## Purpose

This project was built as an interactive learning tool for understanding what backpropagation is actually computing, particularly the relationship between:

- forward propagation
- weighted sums and activations
- loss
- derivatives
- the chain rule
- gradient propagation
- parameter updates
- gradient descent

It is part of a collection of interactive visualizations for understanding deep learning algorithms and papers.
