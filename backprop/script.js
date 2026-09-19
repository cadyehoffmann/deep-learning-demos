"use strict";

/* ================================================================
   BACKPROPAGATION EXPLORER

   Network:
       x1 ──w1──> h1 ──w5──\
        \        /          \
         \w3   /             > yHat -> Loss
         /    \             /
        /      \           /
       x2 ──w2──> h1      /
        \                /
         \w4──> h2 ──w6─/

   Hidden activation: sigmoid
   Output activation: sigmoid

   Loss:
       L = 1/2 (yHat - target)^2

   Biases are included in the mathematics but are not drawn as
   separate edges, keeping the visual network readable.
   ================================================================ */


/* ================================================================
   MODEL STATE
   ================================================================ */

const INITIAL_WEIGHTS = {
  w1: 0.50,   // x1 -> h1
  w2: -0.40,  // x2 -> h1

  w3: 0.30,   // x1 -> h2
  w4: 0.80,   // x2 -> h2

  w5: -0.70,  // h1 -> output
  w6: 0.60,   // h2 -> output

  b1: 0.10,
  b2: -0.20,
  b3: 0.05
};


let weights = { ...INITIAL_WEIGHTS };

let values = {};

let gradients = {};

let trainingSteps = 0;

let selectedWeight = null;


/*
   Controls which calculated values have actually been revealed
   to the user during the animated forward pass.

   The entire forward pass can be computed internally, but the
   visualization should only reveal values when their step occurs.
*/

let revealed = {
  h1: false,
  h2: false,
  yHat: false,
  loss: false
};


/*
   phase:
   "ready"
   "forward"
   "forward-complete"
   "backward"
   "backward-complete"
   "updated"
*/

let phase = "ready";

let forwardStep = 0;
let backwardStep = 0;

let isTraining = false;

let selectedTrainingSteps = 1;


/* ================================================================
   DOM HELPERS
   ================================================================ */

const $ = (id) => document.getElementById(id);


const x1Input = $("x1");
const x2Input = $("x2");
const targetInput = $("target");
const learningRateInput = $("learningRate");
const b1Input = $("b1");
const b2Input = $("b2");
const b3Input = $("b3");

const resetBtn = $("resetBtn");
const forwardBtn = $("forwardBtn");
const backwardBtn = $("backwardBtn");
const updateBtn = $("updateBtn");

const train1Btn = $("train1Btn");
const train25Btn = $("train25Btn");
const train100Btn = $("train100Btn");


/* ================================================================
   BASIC MATH
   ================================================================ */

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}


function sigmoidPrimeFromOutput(output) {
  return output * (1 - output);
}

function frac(top, bottom) {
  return `
    <span class="math-frac">
      <span class="num">${top}</span>
      <span class="den">${bottom}</span>
    </span>
  `;
}

function fmt(value, digits = 4) {

  if (value === undefined || value === null || Number.isNaN(value)) {
    return "—";
  }

  if (Math.abs(value) < 0.00005) {
    return "0.0000";
  }

  return Number(value).toFixed(digits);
}

function syncBiasInputsFromWeights() {

  b1Input.value =
    weights.b1.toFixed(4);

  b2Input.value =
    weights.b2.toFixed(4);

  b3Input.value =
    weights.b3.toFixed(4);
}

function biasChanged() {

  weights.b1 =
    Number(b1Input.value);

  weights.b2 =
    Number(b2Input.value);

  weights.b3 =
    Number(b3Input.value);

  inputChanged();
}


/* ================================================================
   FORWARD PASS
   ================================================================ */

function computeForward() {

  const x1 = Number(x1Input.value);
  const x2 = Number(x2Input.value);
  const target = Number(targetInput.value);

  /*
      Hidden neuron 1

      z1 = w1*x1 + w2*x2 + b1
      h1 = sigmoid(z1)
  */

  const z1 =
    weights.w1 * x1 +
    weights.w2 * x2 +
    weights.b1;

  const h1 = sigmoid(z1);


  /*
      Hidden neuron 2
  */

  const z2 =
    weights.w3 * x1 +
    weights.w4 * x2 +
    weights.b2;

  const h2 = sigmoid(z2);


  /*
      Output neuron
  */

  const z3 =
    weights.w5 * h1 +
    weights.w6 * h2 +
    weights.b3;

  const yHat = sigmoid(z3);


  /*
      Squared error

      L = 1/2 (yHat - target)^2
  */

  const loss =
    0.5 * Math.pow(yHat - target, 2);


  values = {
    x1,
    x2,
    target,

    z1,
    h1,

    z2,
    h2,

    z3,
    yHat,

    loss
  };

  return values;
}


/* ================================================================
   BACKWARD PASS
   ================================================================ */

function computeBackward() {

  /*
      Start at:

          L = 1/2 (yHat - target)^2

      Therefore:

          dL/dyHat = yHat - target
  */

  const dL_dyHat =
    values.yHat - values.target;


  /*
      yHat = sigmoid(z3)

      dyHat/dz3 = yHat(1-yHat)

      Chain rule:

      dL/dz3 =
          dL/dyHat *
          dyHat/dz3
  */

  const dyHat_dz3 =
    sigmoidPrimeFromOutput(values.yHat);

  const dL_dz3 =
    dL_dyHat * dyHat_dz3;


  /*
      z3 = w5*h1 + w6*h2 + b3

      Local derivatives:

      dz3/dw5 = h1
      dz3/dw6 = h2

      Therefore:

      dL/dw5 = dL/dz3 * h1
      dL/dw6 = dL/dz3 * h2
  */

  const dL_dw5 =
    dL_dz3 * values.h1;

  const dL_dw6 =
    dL_dz3 * values.h2;


  /*
      Gradient flowing into hidden activations.

      dz3/dh1 = w5
      dz3/dh2 = w6
  */

  const dL_dh1 =
    dL_dz3 * weights.w5;

  const dL_dh2 =
    dL_dz3 * weights.w6;


  /*
      Hidden activations:

      h1 = sigmoid(z1)
      h2 = sigmoid(z2)
  */

  const dh1_dz1 =
    sigmoidPrimeFromOutput(values.h1);

  const dh2_dz2 =
    sigmoidPrimeFromOutput(values.h2);


  const dL_dz1 =
    dL_dh1 * dh1_dz1;

  const dL_dz2 =
    dL_dh2 * dh2_dz2;


  /*
      z1 = w1*x1 + w2*x2 + b1

      dL/dw1 = dL/dz1 * x1
      dL/dw2 = dL/dz1 * x2
  */

  const dL_dw1 =
    dL_dz1 * values.x1;

  const dL_dw2 =
    dL_dz1 * values.x2;


  /*
      z2 = w3*x1 + w4*x2 + b2
  */

  const dL_dw3 =
    dL_dz2 * values.x1;

  const dL_dw4 =
    dL_dz2 * values.x2;


  /*
      Gradients with respect to the INPUTS.

      This is a nice example of gradient accumulation.

      x1 affects the loss through TWO paths:

          x1 -> h1 -> yHat -> L
          x1 -> h2 -> yHat -> L

      Therefore the contributions ADD.
  */

  const dL_dx1_from_h1 =
    dL_dz1 * weights.w1;

  const dL_dx1_from_h2 =
    dL_dz2 * weights.w3;

  const dL_dx1 =
    dL_dx1_from_h1 +
    dL_dx1_from_h2;


  const dL_dx2_from_h1 =
    dL_dz1 * weights.w2;

  const dL_dx2_from_h2 =
    dL_dz2 * weights.w4;

  const dL_dx2 =
    dL_dx2_from_h1 +
    dL_dx2_from_h2;


  gradients = {

    dL_dyHat,
    dyHat_dz3,
    dL_dz3,

    dL_dw5,
    dL_dw6,

    dL_dh1,
    dL_dh2,

    dh1_dz1,
    dh2_dz2,

    dL_dz1,
    dL_dz2,

    dL_dw1,
    dL_dw2,
    dL_dw3,
    dL_dw4,

    dL_db1: dL_dz1,
    dL_db2: dL_dz2,
    dL_db3: dL_dz3,

    dL_dx1_from_h1,
    dL_dx1_from_h2,
    dL_dx1,

    dL_dx2_from_h1,
    dL_dx2_from_h2,
    dL_dx2
  };

  return gradients;
}


/* ================================================================
   FORWARD ANIMATION STEPS
   ================================================================ */

const forwardSteps = [

  {
    title: "Read the inputs",

    description:
      "The input values enter the network. Nothing has been computed yet.",

    nodes: [
      "node-x1",
      "node-x2"
    ],

    edges: [],

    calculationTitle:
      "Input values",

    formula: () =>
        `
        The forward pass starts with the two input values:

        <br><br>

        x₁ = ${fmt(values.x1)}
        &nbsp;&nbsp;&nbsp;
        x₂ = ${fmt(values.x2)}

        <br><br>

        These values have not been transformed yet.

        <br><br>

        Each input will travel along a weighted connection to both
        hidden neurons, h₁ and h₂.
        `,

    details:
      "These inputs will feed into both hidden neurons.",

    networkLine1: () =>
      `x₁ = ${fmt(values.x1)}     x₂ = ${fmt(values.x2)}`,

    networkLine2: () =>
      ``
  },


  {
    title: "Compute hidden neuron h₁",

    description:
      "First compute the weighted sum z₁. Then pass z₁ through the sigmoid activation function.",

    nodes: [
      "node-h1"
    ],

    edges: [
      "edge-w1",
      "edge-w2"
    ],

    calculationTitle:
      "Hidden neuron h₁",

    formula: () =>
        `
        x₁, x₂ → z₁ → h₁

        <br><br>

        First, combine the inputs using the weights leading into h₁
        and add the bias b₁:

        <br><br>

        z₁ = w₁x₁ + w₂x₂ + b₁

        <br><br>

        Substitute the current values:

        <br><br>

        z₁ =
        (${fmt(weights.w1)})(${fmt(values.x1)})
        +
        (${fmt(weights.w2)})(${fmt(values.x2)})
        +
        ${fmt(weights.b1)}

        <br><br>

        = ${fmt(values.z1)}

        <br><br><br>

        z₁ is the weighted sum entering the neuron.
        It is not yet the output of h₁.

        <br><br>

        Now pass z₁ through the sigmoid activation function:

        <br><br>

        h₁ = σ(z₁)= σ(${fmt(values.z1)}) = ${fmt(values.h1)}

        <br><br>

        This is the value h₁ sends forward to the next layer.
        `,

    details:
      "The bias b₁ is a learnable parameter added to the weighted sum. The weighted sum z₁ is not yet the hidden-neuron output. The sigmoid transforms z₁ into the activation h₁.",

    networkLine1: () =>
        `z₁ = w₁x₁ + w₂x₂ + b₁ = ${fmt(values.z1)}`,

    networkLine2: () =>
        `h₁ = σ(z₁) = σ(${fmt(values.z1)}) = ${fmt(values.h1)}`
  },


  {
    title: "Compute hidden neuron h₂",

    description:
      "Now perform the same two operations for the second hidden neuron.",

    nodes: [
      "node-h2"
    ],

    edges: [
      "edge-w3",
      "edge-w4"
    ],

    calculationTitle:
      "Hidden neuron h₂",

    formula: () =>
        `
        x₁, x₂ → z₂ → h₂

        <br><br>

        Now do the same thing for the second hidden neuron.

        <br><br>

        First, combine the inputs using the weights leading into h₂
        and add the bias b₂:

        <br><br>

        z₂ = w₃x₁ + w₄x₂ + b₂

        <br><br>

        Substitute the current values:

        <br><br>

        z₂ =
        (${fmt(weights.w3)})(${fmt(values.x1)})
        +
        (${fmt(weights.w4)})(${fmt(values.x2)})
        +
        ${fmt(weights.b2)}

        <br><br>

        = ${fmt(values.z2)}

        <br><br><br>

        z₂ is the weighted sum entering the second hidden neuron.

        <br><br>

        Now pass z₂ through the sigmoid:

        <br><br>

        h₂ = σ(z₂) = σ(${fmt(values.z2)}) = ${fmt(values.h2)}

        <br><br>

        This is the value h₂ sends forward to the next layer.
        `,

            details:
            "h₁ and h₂ use the same activation function, but different weights and biases allow them to learn different features.",

            networkLine1: () =>
                `z₂ = w₃x₁ + w₄x₂ + b₂ = ${fmt(values.z2)}`,

            networkLine2: () =>
                `h₂ = σ(z₂) = σ(${fmt(values.z2)}) = ${fmt(values.h2)}`
        },


  {
    title: "Compute the network prediction",

    description:
      "The two hidden activations feed into the output neuron. It also computes a weighted sum followed by sigmoid.",

    nodes: [
      "node-yhat"
    ],

    edges: [
      "edge-w5",
      "edge-w6"
    ],

    calculationTitle:
      "Output neuron",

    formula: () =>
        `
        h₁, h₂ → z₃ → ŷ

        <br><br>

        The output neuron receives h₁ and h₂ as its inputs.

        <br><br>

        First, combine those hidden activations using the output
        weights w₅ and w₆, then add b₃:

        <br><br>

        z₃ = w₅h₁ + w₆h₂ + b₃

        <br><br>

        Substitute the current values:

        <br><br>

        z₃ =
        (${fmt(weights.w5)})(${fmt(values.h1)})
        +
        (${fmt(weights.w6)})(${fmt(values.h2)})
        +
        ${fmt(weights.b3)}

        <br><br>

        = ${fmt(values.z3)}

        <br><br><br>

        z₃ is the weighted sum entering the output neuron.

        <br><br>

        Now pass z₃ through the sigmoid to turn it into the
        network's prediction:

        <br><br>

        ŷ = σ(z₃) = σ(${fmt(values.z3)}) = ${fmt(values.yHat)}

        <br><br>

        ŷ is the network's final prediction.
        `,

    details:
      "ŷ is the network's prediction.",

    networkLine1: () =>
        `z₃ = w₅h₁ + w₆h₂ + b₃ = ${fmt(values.z3)}`,

    networkLine2: () =>
        `ŷ = σ(z₃) = σ(${fmt(values.z3)}) = ${fmt(values.yHat)}`
  },


  {
    title: "Compute the prediction error",

    description:
      "Finally, compare the prediction ŷ with the target y.",

    nodes: [
      "node-loss"
    ],

    edges: [],

    calculationTitle:
      "Squared-error loss",

    formula: () =>
        `
        ŷ → L

        <br><br>

        We now have the network's prediction ŷ.

        <br><br>

        Compare the prediction with the true target y using
        squared-error loss:

        <br><br>

        L = ½(ŷ − y)²

        <br><br>

        Substitute the prediction and target:

        <br><br>

        L =
        ½(${fmt(values.yHat)} − ${fmt(values.target)})² = ${fmt(values.loss, 6)}

        <br><br>

        The loss is a single number measuring how far the prediction
        is from the target.

        <br><br>

        The forward pass is now complete. Backpropagation will work
        backward from this loss to determine how each earlier value
        contributed to it.
        `,

    details:
      "This single number measures how wrong the prediction is. Backpropagation will ask how every weight contributed to this loss.",

    networkLine1: () =>
        `L = ½(ŷ − y)² = ½(${fmt(values.yHat)} − ${fmt(values.target)})² = ${fmt(values.loss, 6)}`,

    networkLine2: () =>
      ``
  }

];


/* ================================================================
   BACKWARD ANIMATION STEPS
   ================================================================ */

const backwardSteps = [

  {
    title: "Start at the loss",

    description:
      "Backpropagation begins by asking how the loss changes when the prediction changes.",

    nodes: ["node-loss", "node-yhat"],

    edges: [],

    calculationTitle:
      "How sensitive is loss to the prediction?",

    formula: () =>
      `
      ŷ → L

      <br><br>

      We start with the loss function:

      <br><br>

      L = ½(ŷ − y)²

      <br><br>

      We want to know how much L changes when ŷ changes,
      so differentiate L with respect to ŷ:

      <br><br>

      ${frac("∂L", "∂ŷ")}
      =
      ${frac("∂", "∂ŷ")}
      [½(ŷ − y)²]

      <br><br>

      = ½ · 2(ŷ − y)

      <br><br>

      = ŷ − y

      <br><br>

      Now substitute the prediction and target:

      <br><br>

      = ${fmt(values.yHat)} − ${fmt(values.target)}

      <br><br>

      = ${fmt(gradients.dL_dyHat)}

      <br><br>

      This is our starting gradient: how sensitive the loss is
      to a change in the network's prediction.
      `,

    networkLine1: () =>
      `∂L/∂ŷ = ŷ − y`,

    networkLine2: () =>
      `= ${fmt(values.yHat)} − ${fmt(values.target)} = ${fmt(gradients.dL_dyHat)}`,

    upstream: () => 1,

    local: () => gradients.dL_dyHat,

    whole: () => gradients.dL_dyHat,

    explanation:
      "Backpropagation starts at the loss and works backward through every operation that produced it."
  },


  {
    title: "Move through the output sigmoid",

    description:
      "Now trace the gradient backward from ŷ to z₃.",

    nodes: ["node-yhat"],

    edges: [],

    calculationTitle:
      "Chain rule at the output",

    formula: () =>
      `
      z₃ → ŷ → L

      <br><br>

      First, we already know how the loss responds to ŷ:

      <br><br>

      ${frac("∂L", "∂ŷ")}
      =
      ${fmt(gradients.dL_dyHat)}

      <br><br>

      Now find how ŷ responds to z₃.

      <br><br>

      Since ŷ = σ(z₃), the sigmoid derivative is:

      <br><br>

      ${frac("∂ŷ", "∂z₃")}
      =
      ŷ(1 − ŷ)

      <br><br>

      =
      ${fmt(values.yHat)}(1 − ${fmt(values.yHat)})

      <br><br>

      =
      ${fmt(gradients.dyHat_dz3)}

      <br><br><br>

      Because z₃ affects L through ŷ, multiply along the path:

      <br><br>

      ${frac("∂L", "∂z₃")}
      =
      ${frac("∂L", "∂ŷ")}
      ${frac("∂ŷ", "∂z₃")}

      <br><br>

      =
      (${fmt(gradients.dL_dyHat)})
      (${fmt(gradients.dyHat_dz3)})

      <br><br>

      =
      ${fmt(gradients.dL_dz3)}
      `,

    networkLine1: () =>
      `∂ŷ/∂z₃ = ŷ(1 − ŷ) = ${fmt(gradients.dyHat_dz3)}`,

    networkLine2: () =>
      `∂L/∂z₃ = ${fmt(gradients.dL_dyHat)} × ${fmt(gradients.dyHat_dz3)} = ${fmt(gradients.dL_dz3)}`,

    upstream: () => gradients.dL_dyHat,

    local: () => gradients.dyHat_dz3,

    whole: () => gradients.dL_dz3,

    explanation:
      "Along a single path, backpropagation multiplies the incoming gradient by the local derivative."
  },


  {
    title: "Find the gradients for w₅ and w₆",

    description:
      "Now trace the gradient from z₃ backward to the two output weights.",

    nodes: ["node-yhat"],

    edges: ["edge-w5", "edge-w6"],

    calculationTitle:
      "Output-layer weight gradients",

    formula: () =>
        `
        w₅ → z₃ → ŷ → L
        &nbsp;&nbsp;&nbsp;
        w₆ → z₃ → ŷ → L

        <br><br>

        We already know how the loss responds to z₃:

        <br><br>

        ${frac("∂L", "∂z₃")}
        =
        ${fmt(gradients.dL_dz3)}

        <br><br>

        Now we want to know how changing w₅ or w₆ changes z₃.

        <br><br>

        Recall the equation for z₃:

        <br><br>

        z₃ = w₅h₁ + w₆h₂ + b₃

        <br><br><br>

        First, consider w₅.

        <br><br>

        w₅ is multiplied by h₁, so the derivative of z₃
        with respect to w₅ is h₁:

        <br><br>

        ${frac("∂z₃", "∂w₅")}
        =
        h₁

        <br><br>

        = ${fmt(values.h1)}

        <br><br>

        Because w₅ affects L through z₃, multiply along the path:

        <br><br>

        ${frac("∂L", "∂w₅")}
        =
        ${frac("∂L", "∂z₃")}
        ${frac("∂z₃", "∂w₅")}

        <br><br>

        =
        (${fmt(gradients.dL_dz3)})
        (${fmt(values.h1)})

        <br><br>

        =
        ${fmt(gradients.dL_dw5)}

        <br><br><br>

        Now do the same thing for w₆.

        <br><br>

        w₆ is multiplied by h₂, so:

        <br><br>

        ${frac("∂z₃", "∂w₆")}
        =
        h₂

        <br><br>

        = ${fmt(values.h2)}

        <br><br>

        Because w₆ also affects L through z₃, multiply along its path:

        <br><br>

        ${frac("∂L", "∂w₆")}
        =
        ${frac("∂L", "∂z₃")}
        ${frac("∂z₃", "∂w₆")}

        <br><br>

        =
        (${fmt(gradients.dL_dz3)})
        (${fmt(values.h2)})

        <br><br>

        =
        ${fmt(gradients.dL_dw6)}
        `,

    networkLine1: () =>
      `∂L/∂w₅ = ${fmt(gradients.dL_dz3)} × ${fmt(values.h1)} = ${fmt(gradients.dL_dw5)}`,

    networkLine2: () =>
      `∂L/∂w₆ = ${fmt(gradients.dL_dz3)} × ${fmt(values.h2)} = ${fmt(gradients.dL_dw6)}`,

    upstream: () => gradients.dL_dz3,

    local: () => values.h1,

    whole: () => gradients.dL_dw5,

    explanation:
      "For a weighted connection, the derivative with respect to the weight is the value entering that connection."
  },


  {
    title: "Send gradients into the hidden layer",

    description:
      "Now trace the loss backward from z₃ to h₁ and h₂.",

    nodes: ["node-h1", "node-h2"],

    edges: ["edge-w5", "edge-w6"],

    calculationTitle:
      "Gradient arriving at the hidden neurons",

    formula: () =>
        `
        h₁ → z₃ → ŷ → L
        &nbsp;&nbsp;&nbsp;
        h₂ → z₃ → ŷ → L

        <br><br>

        We have found how the output weights affect the loss.

        <br><br>

        Now we want to move the gradient farther backward and ask:
        how does the loss respond when h₁ or h₂ changes?

        <br><br>

        We already know:

        <br><br>

        ${frac("∂L", "∂z₃")}
        =
        ${fmt(gradients.dL_dz3)}

        <br><br>

        Recall:

        <br><br>

        z₃ = w₅h₁ + w₆h₂ + b₃

        <br><br><br>

        First, consider h₁.

        <br><br>

        h₁ is multiplied by w₅, so changing h₁ changes z₃
        according to w₅:

        <br><br>

        ${frac("∂z₃", "∂h₁")}
        =
        w₅

        <br><br>

        = ${fmt(weights.w5)}

        <br><br>

        Because h₁ affects L through z₃, multiply along the path:

        <br><br>

        ${frac("∂L", "∂h₁")}
        =
        ${frac("∂L", "∂z₃")}
        ${frac("∂z₃", "∂h₁")}

        <br><br>

        =
        (${fmt(gradients.dL_dz3)})
        (${fmt(weights.w5)})

        <br><br>

        =
        ${fmt(gradients.dL_dh1)}

        <br><br><br>

        Now consider h₂.

        <br><br>

        h₂ is multiplied by w₆, so:

        <br><br>

        ${frac("∂z₃", "∂h₂")}
        =
        w₆

        <br><br>

        = ${fmt(weights.w6)}

        <br><br>

        Because h₂ affects L through z₃, multiply along its path:

        <br><br>

        ${frac("∂L", "∂h₂")}
        =
        ${frac("∂L", "∂z₃")}
        ${frac("∂z₃", "∂h₂")}

        <br><br>

        =
        (${fmt(gradients.dL_dz3)})
        (${fmt(weights.w6)})

        <br><br>

        =
        ${fmt(gradients.dL_dh2)}

        <br><br>

        The gradient has now reached the two hidden activations.
        `,

    networkLine1: () =>
      `∂L/∂h₁ = ${fmt(gradients.dL_dz3)} × ${fmt(weights.w5)} = ${fmt(gradients.dL_dh1)}`,

    networkLine2: () =>
      `∂L/∂h₂ = ${fmt(gradients.dL_dz3)} × ${fmt(weights.w6)} = ${fmt(gradients.dL_dh2)}`,

    upstream: () => gradients.dL_dz3,

    local: () => weights.w5,

    whole: () => gradients.dL_dh1,

    explanation:
      "The gradient has now moved backward through the output connections and reached the hidden activations."
  },


  {
    title: "Move through the hidden sigmoids",

    description:
      "Continue backward through the sigmoid inside each hidden neuron.",

    nodes: ["node-h1", "node-h2"],

    edges: [],

    calculationTitle:
      "Hidden-layer chain rule",

    formula: () =>
        `
        z₁ → h₁ → ... → L
        &nbsp;&nbsp;&nbsp;
        z₂ → h₂ → ... → L

        <br><br>

        We now know how the loss responds to h₁ and h₂.

        <br><br>

        But h₁ and h₂ were produced by applying sigmoid to
        z₁ and z₂, so we need to move backward through those
        sigmoid operations.

        <br><br><br>

        First, for h₁ = σ(z₁):

        <br><br>

        The local sigmoid derivative is:

        <br><br>

        ${frac("∂h₁", "∂z₁")}
        =
        h₁(1 − h₁)

        <br><br>

        =
        ${fmt(values.h1)}(1 − ${fmt(values.h1)})

        <br><br>

        =
        ${fmt(gradients.dh1_dz1)}

        <br><br>

        Because z₁ affects L through h₁, multiply along the path:

        <br><br>

        ${frac("∂L", "∂z₁")}
        =
        ${frac("∂L", "∂h₁")}
        ${frac("∂h₁", "∂z₁")}

        <br><br>

        =
        (${fmt(gradients.dL_dh1)})
        (${fmt(gradients.dh1_dz1)})

        <br><br>

        =
        ${fmt(gradients.dL_dz1)}

        <br><br><br>

        Now do the same thing for h₂ = σ(z₂):

        <br><br>

        ${frac("∂h₂", "∂z₂")}
        =
        h₂(1 − h₂)

        <br><br>

        =
        ${fmt(values.h2)}(1 − ${fmt(values.h2)})

        <br><br>

        =
        ${fmt(gradients.dh2_dz2)}

        <br><br>

        Because z₂ affects L through h₂, multiply along that path:

        <br><br>

        ${frac("∂L", "∂z₂")}
        =
        ${frac("∂L", "∂h₂")}
        ${frac("∂h₂", "∂z₂")}

        <br><br>

        =
        (${fmt(gradients.dL_dh2)})
        (${fmt(gradients.dh2_dz2)})

        <br><br>

        =
        ${fmt(gradients.dL_dz2)}

        <br><br>

        We have now moved the gradient backward through both
        hidden activation functions.
        `,

    networkLine1: () =>
      `∂L/∂z₁ = ${fmt(gradients.dL_dh1)} × ${fmt(gradients.dh1_dz1)} = ${fmt(gradients.dL_dz1)}`,

    networkLine2: () =>
      `∂L/∂z₂ = ${fmt(gradients.dL_dh2)} × ${fmt(gradients.dh2_dz2)} = ${fmt(gradients.dL_dz2)}`,

    upstream: () => gradients.dL_dh1,

    local: () => gradients.dh1_dz1,

    whole: () => gradients.dL_dz1,

    explanation:
      "This is the same operation we used at the output sigmoid: incoming gradient × local sigmoid derivative."
  },


  {
    title: "Find the hidden-layer weight gradients",

    description:
      "Backpropagation has now reached the weights connecting the inputs to the hidden layer.",

    nodes: ["node-h1", "node-h2"],

    edges: [
      "edge-w1",
      "edge-w2",
      "edge-w3",
      "edge-w4"
    ],

    calculationTitle:
      "Hidden-layer weight gradients",

    formula: () =>
        `
        We have now propagated the gradient back to z₁ and z₂.

        <br><br>

        Next, we want to know how each of the four first-layer
        weights affects the loss.

        <br><br><br>

        Start with:

        <br><br>

        z₁ = w₁x₁ + w₂x₂ + b₁

        <br><br>

        First consider w₁.

        <br><br>

        w₁ is multiplied by x₁, so:

        <br><br>

        ${frac("∂z₁", "∂w₁")}
        =
        x₁
        =
        ${fmt(values.x1)}

        <br><br>

        Because w₁ affects L through z₁, multiply along the path:

        <br><br>

        ${frac("∂L", "∂w₁")}
        =
        ${frac("∂L", "∂z₁")}
        ${frac("∂z₁", "∂w₁")}

        <br><br>

        =
        (${fmt(gradients.dL_dz1)})
        (${fmt(values.x1)})

        <br><br>

        =
        ${fmt(gradients.dL_dw1)}

        <br><br><br>

        Now consider w₂.

        <br><br>

        w₂ is multiplied by x₂, so:

        <br><br>

        ${frac("∂z₁", "∂w₂")}
        =
        x₂
        =
        ${fmt(values.x2)}

        <br><br>

        Therefore:

        <br><br>

        ${frac("∂L", "∂w₂")}
        =
        ${frac("∂L", "∂z₁")}
        ${frac("∂z₁", "∂w₂")}

        <br><br>

        =
        (${fmt(gradients.dL_dz1)})
        (${fmt(values.x2)})

        <br><br>

        =
        ${fmt(gradients.dL_dw2)}

        <br><br><br>

        Now move to the second hidden neuron:

        <br><br>

        z₂ = w₃x₁ + w₄x₂ + b₂

        <br><br>

        w₃ is multiplied by x₁, so:

        <br><br>

        ${frac("∂z₂", "∂w₃")}
        =
        x₁
        =
        ${fmt(values.x1)}

        <br><br>

        Therefore:

        <br><br>

        ${frac("∂L", "∂w₃")}
        =
        ${frac("∂L", "∂z₂")}
        ${frac("∂z₂", "∂w₃")}

        <br><br>

        =
        (${fmt(gradients.dL_dz2)})
        (${fmt(values.x1)})

        <br><br>

        =
        ${fmt(gradients.dL_dw3)}

        <br><br><br>

        Finally, w₄ is multiplied by x₂:

        <br><br>

        ${frac("∂z₂", "∂w₄")}
        =
        x₂
        =
        ${fmt(values.x2)}

        <br><br>

        Therefore:

        <br><br>

        ${frac("∂L", "∂w₄")}
        =
        ${frac("∂L", "∂z₂")}
        ${frac("∂z₂", "∂w₄")}

        <br><br>

        =
        (${fmt(gradients.dL_dz2)})
        (${fmt(values.x2)})

        <br><br>

        =
        ${fmt(gradients.dL_dw4)}

        <br><br>

        We now know how sensitive the loss is to every weight
        in the network.
        `,

    networkLine1: () =>
      `∂L/∂w₁ = ${fmt(gradients.dL_dz1)} × ${fmt(values.x1)} = ${fmt(gradients.dL_dw1)}    ∂L/∂w₂ = ${fmt(gradients.dL_dz1)} × ${fmt(values.x2)} = ${fmt(gradients.dL_dw2)}`,

    networkLine2: () =>
      `∂L/∂w₃ = ${fmt(gradients.dL_dz2)} × ${fmt(values.x1)} = ${fmt(gradients.dL_dw3)}    ∂L/∂w₄ = ${fmt(gradients.dL_dz2)} × ${fmt(values.x2)} = ${fmt(gradients.dL_dw4)}`,

    upstream: () => gradients.dL_dz1,

    local: () => values.x1,

    whole: () => gradients.dL_dw1,

    explanation:
      "The gradient has reached the first layer of learnable weights."
  },


  {
    title: "Accumulate gradients where paths meet",

    description:
      "Finally, trace the gradient all the way back to the inputs.",

    nodes: ["node-x1", "node-x2"],

    edges: [
      "edge-w1",
      "edge-w2",
      "edge-w3",
      "edge-w4"
    ],

    calculationTitle:
      "Gradient accumulation",

    formula: () =>
        `
        Each input affects the loss through TWO different paths.

        <br><br>

        This means something new happens here:

        <br><br>

        We multiply derivatives as we move along one path,
        but when multiple paths meet at the same value,
        we ADD their gradient contributions.

        <br><br><br>

        First consider x₁.

        <br><br>

        x₁ reaches the loss through both hidden neurons:

        <br><br>

        x₁ → z₁ → h₁ → ... → L

        <br>

        x₁ → z₂ → h₂ → ... → L

        <br><br>

        First calculate x₁'s contribution through z₁.

        <br><br>

        Because x₁ is multiplied by w₁:

        <br><br>

        ${frac("∂z₁", "∂x₁")}
        =
        w₁
        =
        ${fmt(weights.w1)}

        <br><br>

        So the contribution from this path is:

        <br><br>

        ${frac("∂L", "∂z₁")}
        ${frac("∂z₁", "∂x₁")}

        <br><br>

        =
        (${fmt(gradients.dL_dz1)})
        (${fmt(weights.w1)})

        <br><br>

        =
        ${fmt(gradients.dL_dx1_from_h1)}

        <br><br><br>

        Now calculate x₁'s contribution through z₂.

        <br><br>

        Because x₁ is multiplied by w₃:

        <br><br>

        ${frac("∂z₂", "∂x₁")}
        =
        w₃
        =
        ${fmt(weights.w3)}

        <br><br>

        So the contribution from this path is:

        <br><br>

        ${frac("∂L", "∂z₂")}
        ${frac("∂z₂", "∂x₁")}

        <br><br>

        =
        (${fmt(gradients.dL_dz2)})
        (${fmt(weights.w3)})

        <br><br>

        =
        ${fmt(gradients.dL_dx1_from_h2)}

        <br><br><br>

        Because BOTH paths begin at x₁, add their contributions:

        <br><br>

        ${frac("∂L", "∂x₁")}
        =
        ${fmt(gradients.dL_dx1_from_h1)}
        +
        ${fmt(gradients.dL_dx1_from_h2)}

        <br><br>

        =
        ${fmt(gradients.dL_dx1)}

        <br><br><br>

        The same thing happens for x₂.

        <br><br>

        x₂ reaches the loss through both z₁ and z₂,
        so calculate both path contributions and add them:

        <br><br>

        Through z₁:

        <br><br>

        (${fmt(gradients.dL_dz1)})
        (${fmt(weights.w2)})
        =
        ${fmt(gradients.dL_dx2_from_h1)}

        <br><br>

        Through z₂:

        <br><br>

        (${fmt(gradients.dL_dz2)})
        (${fmt(weights.w4)})
        =
        ${fmt(gradients.dL_dx2_from_h2)}

        <br><br>

        Add the two contributions:

        <br><br>

        ${frac("∂L", "∂x₂")}
        =
        ${fmt(gradients.dL_dx2_from_h1)}
        +
        ${fmt(gradients.dL_dx2_from_h2)}

        <br><br>

        =
        ${fmt(gradients.dL_dx2)}

        <br><br><br>

        Backpropagation has now reached the inputs.

        <br><br>

        The pattern is: multiply derivatives along a path,
        and add gradients wherever multiple paths meet.
        `,

    networkLine1: () =>
      `∂L/∂x₁ = ${fmt(gradients.dL_dx1_from_h1)} + ${fmt(gradients.dL_dx1_from_h2)} = ${fmt(gradients.dL_dx1)}`,

    networkLine2: () =>
      `∂L/∂x₂ = ${fmt(gradients.dL_dx2_from_h1)} + ${fmt(gradients.dL_dx2_from_h2)} = ${fmt(gradients.dL_dx2)}`,

    upstream: () => gradients.dL_dx1_from_h1,

    local: () => 1,

    whole: () => gradients.dL_dx1,

    explanation:
      "Along a path, derivatives multiply. When multiple paths lead back to the same value, their gradient contributions add."
  }

];


/* ================================================================
   DISPLAY HELPERS
   ================================================================ */

function clearHighlights() {

  document.querySelectorAll(".node").forEach(node => {
    node.classList.remove(
      "active-forward",
      "active-backward",
      "active-update"
    );
  });

  document.querySelectorAll(".edge").forEach(edge => {
    edge.classList.remove(
      "forward-active",
      "backward-active",
      "updated"
    );
  });
}


function highlightStep(step, direction) {

  clearHighlights();

  const nodeClass =
    direction === "forward"
      ? "active-forward"
      : "active-backward";

  const edgeClass =
    direction === "forward"
      ? "forward-active"
      : "backward-active";


  step.nodes.forEach(id => {
    $(id)?.classList.add(nodeClass);
  });

  step.edges.forEach(id => {
    $(id)?.classList.add(edgeClass);
  });
}


function setBadge(text, className) {

  const badge = $("phaseBadge");

  badge.textContent = text;

  badge.className =
    `phase-badge ${className}`;
}


function updateNetworkValues() {

  $("value-x1").textContent =
    values.x1 === undefined
      ? "?"
      : fmt(values.x1);

  $("value-x2").textContent =
    values.x2 === undefined
      ? "?"
      : fmt(values.x2);


  $("value-h1").textContent =
    revealed.h1 && values.h1 !== undefined
      ? fmt(values.h1)
      : "?";


  $("value-h2").textContent =
    revealed.h2 && values.h2 !== undefined
      ? fmt(values.h2)
      : "?";


  $("value-yhat").textContent =
    revealed.yHat && values.yHat !== undefined
      ? fmt(values.yHat)
      : "?";

  $("value-h1").textContent =
    revealed.h1 && values.h1 !== undefined
        ? fmt(values.h1)
        : "?";

  $("value-z1").textContent =
    revealed.h1 && values.z1 !== undefined
        ? `z₁ = ${fmt(values.z1)}`
        : "";


  $("value-h2").textContent =
    revealed.h2 && values.h2 !== undefined
        ? fmt(values.h2)
        : "?";

  $("value-z2").textContent =
    revealed.h2 && values.z2 !== undefined
        ? `z₂ = ${fmt(values.z2)}`
        : "";


  $("value-yhat").textContent =
    revealed.yHat && values.yHat !== undefined
        ? fmt(values.yHat)
        : "?";

  $("value-z3").textContent =
    revealed.yHat && values.z3 !== undefined
        ? `z₃ = ${fmt(values.z3)}`
        : "";


  $("value-loss").textContent =
    revealed.loss && values.loss !== undefined
      ? fmt(values.loss, 5)
      : "?";
}


function updateEdgeLabels() {

  $("label-w1").textContent =
    `w₁=${fmt(weights.w1, 3)}`;

  $("label-w2").textContent =
    `w₂=${fmt(weights.w2, 3)}`;

  $("label-w3").textContent =
    `w₃=${fmt(weights.w3, 3)}`;

  $("label-w4").textContent =
    `w₄=${fmt(weights.w4, 3)}`;

  $("label-w5").textContent =
    `w₅=${fmt(weights.w5, 3)}`;

  $("label-w6").textContent =
    `w₆=${fmt(weights.w6, 3)}`;
}


function updateStats() {

  $("predictionStat").textContent =
    values.yHat === undefined
      ? "—"
      : fmt(values.yHat);

  $("targetStat").textContent =
    values.target === undefined
      ? targetInput.value
      : fmt(values.target);

  $("lossStat").textContent =
    values.loss === undefined
      ? "—"
      : fmt(values.loss, 6);

  $("trainingStepsStat").textContent =
    trainingSteps;
}


/* ================================================================
   WEIGHT TABLE
   ================================================================ */

const WEIGHT_META = {

  w1: {
    label: "w₁",
    connection: "x₁ → h₁",
    gradient: "dL_dw1"
  },

  w2: {
    label: "w₂",
    connection: "x₂ → h₁",
    gradient: "dL_dw2"
  },

  w3: {
    label: "w₃",
    connection: "x₁ → h₂",
    gradient: "dL_dw3"
  },

  w4: {
    label: "w₄",
    connection: "x₂ → h₂",
    gradient: "dL_dw4"
  },

  w5: {
    label: "w₅",
    connection: "h₁ → ŷ",
    gradient: "dL_dw5"
  },

  w6: {
    label: "w₆",
    connection: "h₂ → ŷ",
    gradient: "dL_dw6"
  }

};


function renderWeightTable() {

  const tbody = $("weightTable");

  tbody.innerHTML = "";


  Object.entries(WEIGHT_META).forEach(([key, meta]) => {

    const gradient =
      gradients[meta.gradient];

    const pennyEffect =
      gradient === undefined
        ? undefined
        : gradient * 0.01;


    const row =
      document.createElement("tr");

    row.className =
      "weight-row" +
      (selectedWeight === key ? " selected" : "");


    row.innerHTML = `
      <td><strong>${meta.label}</strong></td>
      <td>${meta.connection}</td>
      <td>${fmt(weights[key])}</td>
      <td>${fmt(gradient, 6)}</td>
      <td>${fmt(pennyEffect, 7)}</td>
    `;


    row.addEventListener("click", () => {

      selectedWeight = key;

      renderWeightTable();

      renderPennyExplanation(key);
    });


    tbody.appendChild(row);
  });
}


/* ================================================================
   PENNY INTERPRETATION
   ================================================================ */

function renderPennyExplanation(key) {

  const meta = WEIGHT_META[key];

  const gradient =
    gradients[meta.gradient];


  $("pennyTitle").textContent =
    `${meta.label}: ${meta.connection}`;


  if (gradient === undefined) {

    $("pennyText").textContent =
      "Run the backward pass first. We need ∂L/∂w before we can estimate how a small change in this weight affects the loss.";

    $("pennyFormula").innerHTML =
        `ΔL ≈ ${frac("∂L", "∂w")} Δw`;
    return;
  }


  const deltaW = 0.01;

  const predictedChange =
    gradient * deltaW;


  const direction =
    predictedChange > 0
      ? "increase"
      : predictedChange < 0
        ? "decrease"
        : "barely change";


  $("pennyText").innerHTML =
    `The gradient is <strong>${fmt(gradient, 6)}</strong>.
     If ${meta.label} increases by one penny (+0.01),
     the loss should approximately <strong>${direction}</strong>
     by ${fmt(Math.abs(predictedChange), 7)}.`;


  $("pennyFormula").innerHTML =
    `
    ΔL ≈
    ${frac("∂L", `∂${meta.label}`)}
    Δ${meta.label}

    <br><br>

    ≈ (${fmt(gradient, 6)})(0.01)

    <br><br>

    = ${fmt(predictedChange, 7)}
    `;
}


/* ================================================================
   FORWARD UI
   ================================================================ */

function showForwardStep(index) {

  const step =
    forwardSteps[index];


  /*
     Reveal values only when their computation has actually
     happened in the visual walkthrough.
  */

  if (index === 1) {
    revealed.h1 = true;
  }

  if (index === 2) {
    revealed.h2 = true;
  }

  if (index === 3) {
    revealed.yHat = true;
  }

  if (index === 4) {
    revealed.loss = true;
  }


  updateNetworkValues();


  /*
     Highlight the active portion of the network.
  */

  highlightStep(
    step,
    "forward"
  );


  setBadge(
    "Forward pass",
    "forward"
  );


  $("stepTitle").textContent =
    step.title;


  $("stepDescription").textContent =
    step.description;


  $("calculationTitle").textContent =
    step.calculationTitle;


  $("formula").innerHTML =
    step.formula();


  $("calculationDetails").textContent =
    step.details;


  $("stepCounter").textContent =
    `${index + 1} / ${forwardSteps.length}`;


  /*
     Update the computation that is shown directly
     underneath the network visualization.
  */

  $("networkComputationLine1").textContent =
    step.networkLine1();


  $("networkComputationLine2").textContent =
    step.networkLine2();

    // Position the calculation text inside the gray box above the network.
    // Steps with one line are vertically centered.
    // Steps with two lines use both rows.

    if (
        index === 0 ||
        index === 4
    ) {

        $("networkComputationLine1").setAttribute("y", "55");
        $("networkComputationLine2").setAttribute("y", "55");

      } else {

        $("networkComputationLine1").setAttribute("y", "44");
        $("networkComputationLine2").setAttribute("y", "67");

      }


  /*
     Gradients do not exist yet during the forward pass.
  */

  $("upstreamValue").textContent =
    "—";

  $("localValue").textContent =
    "—";

  $("wholeValue").textContent =
    "—";


  $("gradientExplanation").textContent =
    "Derivative values appear here during the backward pass.";
}


/* ================================================================
   BACKWARD UI
   ================================================================ */

function getGradientDisplay(index) {

  const displays = [

    // ------------------------------------------------------------
    // STEP 1: LOSS -> PREDICTION
    // ------------------------------------------------------------

    {
      upstreamSymbol: `
        Starting gradient<br>
        1
      `,

      localSymbol: `
        ${frac("∂L", "∂ŷ")}
        <br><br>
        = ŷ − y
        <br><br>
        = ${fmt(values.yHat)} − ${fmt(values.target)}
        <br><br>
        = ${fmt(gradients.dL_dyHat)}
      `,

      wholeSymbol: `
        ${frac("∂L", "∂ŷ")}
        <br><br>
        = 1 × ${frac("∂L", "∂ŷ")}
        <br><br>
        = 1 × ${fmt(gradients.dL_dyHat)}
        <br><br>
        = ${fmt(gradients.dL_dyHat)}
      `,

      forwardPath:
        "ŷ → L",

      backwardPath:
        "L → ŷ"
    },


    // ------------------------------------------------------------
    // STEP 2: OUTPUT SIGMOID
    // ------------------------------------------------------------

    {
      upstreamSymbol: `
        ${frac("∂L", "∂ŷ")}
        <br><br>
        = ŷ − y
        <br><br>
        = ${fmt(values.yHat)} − ${fmt(values.target)}
        <br><br>
        = ${fmt(gradients.dL_dyHat)}
      `,

      localSymbol: `
        ${frac("∂ŷ", "∂z₃")}
        <br><br>
        = ŷ(1 − ŷ)
        <br><br>
        = ${fmt(values.yHat)}(1 − ${fmt(values.yHat)})
        <br><br>
        = ${fmt(gradients.dyHat_dz3)}
      `,

      wholeSymbol: `
        ${frac("∂L", "∂z₃")}
        <br><br>
        =
        ${frac("∂L", "∂ŷ")}
        ×
        ${frac("∂ŷ", "∂z₃")}
        <br><br>
        =
        (${fmt(gradients.dL_dyHat)})
        (${fmt(gradients.dyHat_dz3)})
        <br><br>
        =
        ${fmt(gradients.dL_dz3)}
      `,

      forwardPath:
        "z₃ → ŷ → L",

      backwardPath:
        "L → ŷ → z₃"
    },


    // ------------------------------------------------------------
    // STEP 3: OUTPUT WEIGHT w5
    // ------------------------------------------------------------

    {
      upstreamSymbol: `
        ${frac("∂L", "∂z₃")}
        <br><br>
        =
        ${frac("∂L", "∂ŷ")}
        ×
        ${frac("∂ŷ", "∂z₃")}
        <br><br>
        =
        (${fmt(gradients.dL_dyHat)})
        (${fmt(gradients.dyHat_dz3)})
        <br><br>
        =
        ${fmt(gradients.dL_dz3)}
      `,

      localSymbol: `
        ${frac("∂z₃", "∂w₅")}
        <br><br>
        = h₁
        <br><br>
        = ${fmt(values.h1)}
      `,

      wholeSymbol: `
        ${frac("∂L", "∂w₅")}
        <br><br>
        =
        ${frac("∂L", "∂z₃")}
        ×
        ${frac("∂z₃", "∂w₅")}
        <br><br>
        =
        (${fmt(gradients.dL_dz3)})
        (${fmt(values.h1)})
        <br><br>
        =
        ${fmt(gradients.dL_dw5)}
      `,

      forwardPath:
        "w₅ → z₃ → ŷ → L",

      backwardPath:
        "L → ŷ → z₃ → w₅"
    },


    // ------------------------------------------------------------
    // STEP 4: BACK INTO h1
    // ------------------------------------------------------------

    {
      upstreamSymbol: `
        ∂L/∂z₃
        <br>
        = ∂L/∂ŷ × ∂ŷ/∂z₃
        <br>
        = (${fmt(gradients.dL_dyHat)})(${fmt(gradients.dyHat_dz3)})
        <br>
        = ${fmt(gradients.dL_dz3)}
      `,

      localSymbol: `
        ∂z₃/∂h₁
        <br>
        z₃ = w₅h₁ + w₆h₂ + b₃
        <br>
        ∂z₃/∂h₁ = w₅
        <br>
        = ${fmt(weights.w5)}
      `,

      wholeSymbol: `
        ∂L/∂h₁
        <br>
        = ∂L/∂z₃ × ∂z₃/∂h₁
        <br>
        = (${fmt(gradients.dL_dz3)})(${fmt(weights.w5)})
        <br>
        = ${fmt(gradients.dL_dh1)}
      `,

      forwardPath:
        "h₁ → z₃ → ŷ → L",

      backwardPath:
        "L → ŷ → z₃ → h₁"
    },


    // ------------------------------------------------------------
    // STEP 5: HIDDEN SIGMOID
    // ------------------------------------------------------------

    {
      upstreamSymbol: `
        ∂L/∂h₁
        <br>
        = ∂L/∂z₃ × ∂z₃/∂h₁
        <br>
        = (${fmt(gradients.dL_dz3)})(${fmt(weights.w5)})
        <br>
        = ${fmt(gradients.dL_dh1)}
      `,

      localSymbol: `
        ∂h₁/∂z₁
        <br>
        = σ′(z₁)
        <br>
        = ${fmt(values.h1)}(1 − ${fmt(values.h1)})
        <br>
        = ${fmt(gradients.dh1_dz1)}
      `,

      wholeSymbol: `
        ∂L/∂z₁
        <br>
        = ∂L/∂h₁ × ∂h₁/∂z₁
        <br>
        = (${fmt(gradients.dL_dh1)})(${fmt(gradients.dh1_dz1)})
        <br>
        = ${fmt(gradients.dL_dz1)}
      `,

      forwardPath:
        "z₁ → h₁ → z₃ → ŷ → L",

      backwardPath:
        "L → ŷ → z₃ → h₁ → z₁"
    },


    // ------------------------------------------------------------
    // STEP 6: FIRST-LAYER WEIGHT w1
    // ------------------------------------------------------------

    {
      upstreamSymbol: `
        ∂L/∂z₁
        <br>
        = ∂L/∂h₁ × ∂h₁/∂z₁
        <br>
        = (${fmt(gradients.dL_dh1)})(${fmt(gradients.dh1_dz1)})
        <br>
        = ${fmt(gradients.dL_dz1)}
      `,

      localSymbol: `
        ∂z₁/∂w₁
        <br>
        z₁ = w₁x₁ + w₂x₂ + b₁
        <br>
        ∂z₁/∂w₁ = x₁
        <br>
        = ${fmt(values.x1)}
      `,

      wholeSymbol: `
        ∂L/∂w₁
        <br>
        = ∂L/∂z₁ × ∂z₁/∂w₁
        <br>
        = (${fmt(gradients.dL_dz1)})(${fmt(values.x1)})
        <br>
        = ${fmt(gradients.dL_dw1)}
      `,

      forwardPath:
        "w₁ → z₁ → h₁ → z₃ → ŷ → L",

      backwardPath:
        "L → ŷ → z₃ → h₁ → z₁ → w₁"
    },


    // ------------------------------------------------------------
    // STEP 7: GRADIENT ACCUMULATION
    // ------------------------------------------------------------

    {
      upstreamSymbol: `
        Two gradient contributions arrive at x₁.
      `,

      localSymbol: `
        Through h₁:
        ${fmt(gradients.dL_dx1_from_h1)}
        <br><br>
        Through h₂:
        ${fmt(gradients.dL_dx1_from_h2)}
      `,

      wholeSymbol: `
        ∂L/∂x₁
        <br>
        = contribution through h₁
        + contribution through h₂
        <br>
        = ${fmt(gradients.dL_dx1_from_h1)}
        + ${fmt(gradients.dL_dx1_from_h2)}
        <br>
        = ${fmt(gradients.dL_dx1)}
      `,

      forwardPath:
        "x₁ → hidden layer → output → L",

      backwardPath:
        "L → output → hidden layer → x₁"
    }

  ];

  return displays[index];
}

function showBackwardStep(index) {

  const step =
    backwardSteps[index];

  highlightStep(step, "backward");

  setBadge("Backward pass", "backward");

  $("stepTitle").textContent =
    step.title;

  $("stepDescription").textContent =
    step.description;

  $("calculationTitle").textContent =
    step.calculationTitle;

  $("formula").innerHTML =
    step.formula();

  $("calculationDetails").textContent =
    "Follow the red path from the loss back toward the inputs.";

  $("stepCounter").textContent =
    `${index + 1} / ${backwardSteps.length}`;


  // Update the calculation shown directly
  // underneath the network visualization.
  $("networkComputationLine1").textContent =
    step.networkLine1 ? step.networkLine1() : "";

  $("networkComputationLine2").textContent =
    step.networkLine2 ? step.networkLine2() : "";

  $("networkComputationLine1").setAttribute("y", "44");
  $("networkComputationLine2").setAttribute("y", "67");

  const display =
  getGradientDisplay(index);


// Show the extra backprop explanation sections

$("gradientMeaning").hidden = false;
$("gradientPath").hidden = false;


// Show what each gradient represents

$("upstreamSymbol").innerHTML =
  display.upstreamSymbol;

$("localSymbol").innerHTML =
  display.localSymbol;

$("wholeSymbol").innerHTML =
  display.wholeSymbol;


// Show the numerical upstream × local = whole gradient relationship

$("upstreamValue").textContent =
  fmt(step.upstream());

$("localValue").textContent =
  fmt(step.local());

$("wholeValue").textContent =
  fmt(step.whole());


// Show the direction of the computation

$("forwardGradientPath").textContent =
  display.forwardPath;

$("backwardGradientPath").textContent =
  display.backwardPath;


// Explain what is happening at this step

$("gradientExplanation").textContent =
  step.explanation;
}


/* ================================================================
   BUTTON ACTIONS
   ================================================================ */

function stepForward() {

  /*
      Beginning a fresh forward pass.
  */

  if (
    phase === "ready" ||
    phase === "updated" ||
    phase === "backward-complete"
    ) {

    computeForward();

    gradients = {};

    revealed = {
        h1: false,
        h2: false,
        yHat: false,
        loss: false
    };

    forwardStep = 0;

    phase = "forward";

    updateNetworkValues();
    updateStats();
    renderWeightTable();
}


  if (phase !== "forward") {
    return;
  }


  showForwardStep(forwardStep);

  forwardStep++;


  if (forwardStep >= forwardSteps.length) {

    phase = "forward-complete";

    forwardBtn.disabled = true;
    backwardBtn.disabled = false;

    setBadge("Forward pass complete", "forward");

    $("stepTitle").textContent =
      "Prediction complete — now propagate backward";

    $("stepDescription").textContent =
      "The loss has been calculated. Click Step Backward to trace how that loss depends on each weight.";


  }

}


function stepBackward() {

  if (phase === "forward-complete") {

    computeBackward();

    backwardStep = 0;

    phase = "backward";

    renderWeightTable();
  }


  if (phase !== "backward") {
    return;
  }


  showBackwardStep(backwardStep);

  backwardStep++;


  if (backwardStep >= backwardSteps.length) {

    phase = "backward-complete";

    backwardBtn.disabled = true;
    updateBtn.disabled = false;

  }

}


function updateWeights() {

  if (phase !== "backward-complete") {
    return;
  }


  const lr =
    Number(learningRateInput.value);


  /*
      Save the OLD parameter values and network result.

      The gradients we just calculated describe the loss
      at these old parameter values.
  */

  const oldWeights =
    { ...weights };

  const oldPrediction =
    values.yHat;

  const oldLoss =
    values.loss;


  /*
      Gradient descent:

      parameter_new
      =
      parameter_old
      -
      learningRate * gradient
  */

  weights.w1 -= lr * gradients.dL_dw1;
  weights.w2 -= lr * gradients.dL_dw2;

  weights.w3 -= lr * gradients.dL_dw3;
  weights.w4 -= lr * gradients.dL_dw4;

  weights.w5 -= lr * gradients.dL_dw5;
  weights.w6 -= lr * gradients.dL_dw6;

  weights.b1 -= lr * gradients.dL_db1;
  weights.b2 -= lr * gradients.dL_db2;
  weights.b3 -= lr * gradients.dL_db3;


  /*
      Keep the visible bias controls synchronized
      with the newly learned bias values.
  */

  syncBiasInputsFromWeights();


  trainingSteps++;


  /*
      Run a new forward pass using the UPDATED parameters
      so we can see what the gradient-descent step accomplished.
  */

  computeForward();


  phase = "updated";

  forwardStep = 0;
  backwardStep = 0;


  clearHighlights();

  document.querySelectorAll(".edge").forEach(edge => {
    edge.classList.add("updated");
  });


  setBadge(
    "Weights updated",
    "update"
  );


  $("stepTitle").textContent =
    "Use the gradients to update the parameters";

  $("stepDescription").textContent =
    "Gradient descent moves each parameter in the direction that should reduce the loss.";


  $("calculationTitle").textContent =
    "Gradient-descent update";


  $("formula").innerHTML =
    `
    Backpropagation told us how the loss responds to each parameter.

    <br><br>

    A positive gradient means increasing that parameter would increase
    the loss. A negative gradient means increasing it would decrease
    the loss.

    <br><br>

    To move toward a smaller loss, gradient descent moves in the
    <strong>opposite</strong> direction of the gradient:

    <br><br>

    w<sub>new</sub>
    =
    w<sub>old</sub>
    −
    η
    ${frac("∂L", "∂w")}

    <br><br>

    where the learning rate is:

    <br><br>

    η = ${fmt(lr)}

    <br><br><br>

    <strong>Update w₁</strong>

    <br><br>

    w₁,new
    =
    ${fmt(oldWeights.w1)}
    −
    (${fmt(lr)})
    (${fmt(gradients.dL_dw1)})

    <br><br>

    =
    ${fmt(weights.w1)}

    <br><br><br>

    <strong>Update w₂</strong>

    <br><br>

    w₂,new
    =
    ${fmt(oldWeights.w2)}
    −
    (${fmt(lr)})
    (${fmt(gradients.dL_dw2)})

    <br><br>

    =
    ${fmt(weights.w2)}

    <br><br><br>

    <strong>Update w₃</strong>

    <br><br>

    w₃,new
    =
    ${fmt(oldWeights.w3)}
    −
    (${fmt(lr)})
    (${fmt(gradients.dL_dw3)})

    <br><br>

    =
    ${fmt(weights.w3)}

    <br><br><br>

    <strong>Update w₄</strong>

    <br><br>

    w₄,new
    =
    ${fmt(oldWeights.w4)}
    −
    (${fmt(lr)})
    (${fmt(gradients.dL_dw4)})

    <br><br>

    =
    ${fmt(weights.w4)}

    <br><br><br>

    <strong>Update w₅</strong>

    <br><br>

    w₅,new
    =
    ${fmt(oldWeights.w5)}
    −
    (${fmt(lr)})
    (${fmt(gradients.dL_dw5)})

    <br><br>

    =
    ${fmt(weights.w5)}

    <br><br><br>

    <strong>Update w₆</strong>

    <br><br>

    w₆,new
    =
    ${fmt(oldWeights.w6)}
    −
    (${fmt(lr)})
    (${fmt(gradients.dL_dw6)})

    <br><br>

    =
    ${fmt(weights.w6)}

    <br><br><br>

    The biases are trainable parameters too, so they are updated
    using exactly the same rule:

    <br><br>

    b₁,new
    =
    ${fmt(oldWeights.b1)}
    −
    (${fmt(lr)})
    (${fmt(gradients.dL_db1)})
    =
    ${fmt(weights.b1)}

    <br><br>

    b₂,new
    =
    ${fmt(oldWeights.b2)}
    −
    (${fmt(lr)})
    (${fmt(gradients.dL_db2)})
    =
    ${fmt(weights.b2)}

    <br><br>

    b₃,new
    =
    ${fmt(oldWeights.b3)}
    −
    (${fmt(lr)})
    (${fmt(gradients.dL_db3)})
    =
    ${fmt(weights.b3)}

    <br><br><br>

    Now run the network forward again using the new parameters.

    <br><br>

    Old prediction:
    ŷ = ${fmt(oldPrediction)}

    <br>

    New prediction:
    ŷ = ${fmt(values.yHat)}

    <br><br>

    Old loss:
    L = ${fmt(oldLoss, 6)}

    <br>

    New loss:
    L = ${fmt(values.loss, 6)}

    <br><br>

    The parameter update changed the network's prediction and,
    for this step, moved the loss downward.
    `;


  $("calculationDetails").textContent =
    "Backpropagation calculates the gradients; gradient descent uses those gradients to actually change the trainable parameters.";


  $("stepCounter").textContent =
    "✓";


  /*
      Show a concise version directly underneath
      the network visualization.
  */

  $("networkComputationLine1").innerHTML =
    `w<tspan baseline-shift="sub" font-size="75%">new</tspan> = w<tspan baseline-shift="sub" font-size="75%">old</tspan> − η(∂L/∂w)`;
  $("networkComputationLine2").textContent =
    `Loss: ${fmt(oldLoss, 6)} → ${fmt(values.loss, 6)}`;

  $("networkComputationLine1").setAttribute(
    "y",
    "44"
  );

  $("networkComputationLine2").setAttribute(
    "y",
    "67"
  );


  /*
      The backward-pass derivative cards no longer represent
      an active chain-rule step.
  */

  $("upstreamValue").textContent =
    "—";

  $("localValue").textContent =
    "—";

  $("wholeValue").textContent =
    "—";

  $("gradientExplanation").textContent =
    "The backward pass produced the gradients above. Gradient descent now uses them to update the parameters.";


updateNetworkValues();
updateEdgeLabels();
updateStats();
renderWeightTable();


/*
    One complete Forward → Backward → Update cycle
    has now finished.
*/

if (selectedTrainingSteps === 1) {

  setBadge(
    "Training complete",
    "update"
  );

  $("stepTitle").textContent =
    "Completed 1 training step";

  $("stepDescription").textContent =
    "You completed one full learning cycle: forward pass, backpropagation, and parameter update.";

  $("stepCounter").textContent =
    "1 / 1";

} else {

  setBadge(
    `Training 1 / ${selectedTrainingSteps}`,
    "update"
  );

  $("stepTitle").textContent =
    `Completed training step 1 of ${selectedTrainingSteps}`;

  $("stepDescription").textContent =
    "One full learning cycle is complete. The remaining iterations can now repeat the same forward → backward → update process.";

  $("stepCounter").textContent =
    `1 / ${selectedTrainingSteps}`;

  /*
    Step 1 was completed manually.

    If the user selected Train 25 or Train 100,
    automatically continue with the remaining steps.
  */

  setTimeout(() => {
    continueAutomaticTraining(selectedTrainingSteps);
  }, 500);

}


    forwardBtn.disabled = false;
    backwardBtn.disabled = true;
    updateBtn.disabled = true;
}


/* ================================================================
   AUTOMATIC TRAINING
   ================================================================ */

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


function setActiveTrainingButton(numberOfSteps) {

  const trainingTableWrapper =
    document.querySelector(".training-table-wrapper");

  if (trainingTableWrapper) {
    trainingTableWrapper.classList.toggle(
        "scroll-training",
        numberOfSteps === 100
    );
  }
  selectedTrainingSteps = numberOfSteps;

  const buttons = [
    [train1Btn, 1],
    [train25Btn, 25],
    [train100Btn, 100]
  ];

  buttons.forEach(([button, steps]) => {

    button.classList.toggle(
      "secondary",
      steps !== numberOfSteps
    );

  });


  /*
      Train 1 is the manual walkthrough mode.

      If the user returns to it after automatic training,
      remove the old training history and restore the
      Upstream × Local explanation panel.
  */

  if (numberOfSteps === 1) {

    clearTrainingHistory();
    showBackpropIdea();

  }

}


function showTrainingHistory() {

  $("backpropIdeaView").hidden = true;
  $("trainingHistoryView").hidden = false;
}


function showBackpropIdea() {

  $("trainingHistoryView").hidden = true;
  $("backpropIdeaView").hidden = false;
}


function clearTrainingHistory() {

  $("trainingHistoryBody").innerHTML = "";
  $("trainingProgress").textContent = "0 / 0";
}


function addTrainingHistoryRow(iteration) {

  const row = document.createElement("tr");

  row.innerHTML = `
    <td>${iteration}</td>

    <td>${fmt(values.yHat)}</td>
    <td>${fmt(values.loss, 6)}</td>

    <td>${fmt(weights.w1)}</td>
    <td>${fmt(weights.w2)}</td>
    <td>${fmt(weights.w3)}</td>
    <td>${fmt(weights.w4)}</td>
    <td>${fmt(weights.w5)}</td>
    <td>${fmt(weights.w6)}</td>

    <td>${fmt(weights.b1)}</td>
    <td>${fmt(weights.b2)}</td>
    <td>${fmt(weights.b3)}</td>
  `;

  $("trainingHistoryBody").appendChild(row);

  /*
      Keep the newest iteration visible when the
      history becomes taller than the panel.
  */

  const wrapper =
    document.querySelector(".training-table-wrapper");

  wrapper.scrollTop =
    wrapper.scrollHeight;
}


function setTrainingButtonsDisabled(disabled) {

  train1Btn.disabled = disabled;
  train25Btn.disabled = disabled;
  train100Btn.disabled = disabled;
}

async function continueAutomaticTraining(numberOfSteps) {

  if (isTraining) {
    return;
  }

  isTraining = true;

  const lr =
    Number(learningRateInput.value);


  /*
      Switch the right-hand explanation panel
      from manual backprop to training history.
  */

  showTrainingHistory();
  clearTrainingHistory();


  /*
      Step 1 was the manual iteration the user
      just completed.

      Record the current network state as row 1.
  */

  computeForward();

  addTrainingHistoryRow(1);

  $("trainingProgress").textContent =
    `1 / ${numberOfSteps}`;


  revealed = {
    h1: true,
    h2: true,
    yHat: true,
    loss: true
  };


  /*
      Disable controls while automatic training
      finishes the remaining iterations.
  */

  forwardBtn.disabled = true;
  backwardBtn.disabled = true;
  updateBtn.disabled = true;
  resetBtn.disabled = true;

  setTrainingButtonsDisabled(true);


  $("trainingExplanation").innerHTML =
    `
    <strong>Step 1 was completed manually.</strong>

    The demo is now repeating the same learning cycle automatically.

    Each new row records the network after one complete
    forward → backward → update iteration.
    `;
  /*
    During repeated training, show the general
    gradient-descent rule rather than numbers from
    one particular iteration.
*/

$("calculationTitle").textContent =
  "Gradient-descent update rule";


$("formula").innerHTML =
  `
  Every training iteration uses the same update rule:

  <br><br>

  w<sub>new</sub>
  =
  w<sub>old</sub>
  −
  η
  ${frac("∂L", "∂w")}

  <br><br>

  where the learning rate is:

  <br><br>

  η = ${fmt(lr)}

  <br><br><br>

  <strong>Weight updates</strong>

  <br><br>

  w<sub>1</sub><sub>new</sub>
  =
  w<sub>1</sub><sub>old</sub>
  − η ${frac("∂L", "∂w₁")}

  <br><br>

  w<sub>2</sub><sub>new</sub>
  =
  w<sub>2</sub><sub>old</sub>
  − η ${frac("∂L", "∂w₂")}

  <br><br>

  w<sub>3</sub><sub>new</sub>
  =
  w<sub>3</sub><sub>old</sub>
  − η ${frac("∂L", "∂w₃")}

  <br><br>

  w<sub>4</sub><sub>new</sub>
  =
  w<sub>4</sub><sub>old</sub>
  − η ${frac("∂L", "∂w₄")}

  <br><br>

  w<sub>5</sub><sub>new</sub>
  =
  w<sub>5</sub><sub>old</sub>
  − η ${frac("∂L", "∂w₅")}

  <br><br>

  w<sub>6</sub><sub>new</sub>
  =
  w<sub>6</sub><sub>old</sub>
  − η ${frac("∂L", "∂w₆")}

  <br><br><br>

  <strong>Bias updates</strong>

  <br><br>

  b<sub>1</sub><sub>new</sub>
  =
  b<sub>1</sub><sub>old</sub>
  − η ${frac("∂L", "∂b₁")}

  <br><br>

  b<sub>2</sub><sub>new</sub>
  =
  b<sub>2</sub><sub>old</sub>
  − η ${frac("∂L", "∂b₂")}

  <br><br>

  b<sub>3</sub><sub>new</sub>
  =
  b<sub>3</sub><sub>old</sub>
  − η ${frac("∂L", "∂b₃")}

  <br><br><br>

  The formula stays the same on every iteration.
  The parameter values and gradients are recalculated
  after each pass through the network.
  `;


$("calculationDetails").textContent =
  "The table on the right shows how the actual parameter values change from one completed training iteration to the next.";
    


  /*
      25 steps should be slow enough to watch.

      100 steps uses a shorter delay so the demo
      does not take too long.
  */

  const delay =
    numberOfSteps === 25 ? 275 : 140;


  /*
      Start at 2 because iteration 1 was already
      completed manually.
  */

  for (let i = 2; i <= numberOfSteps; i++) {

    /* ----------------------------
       FORWARD
       ---------------------------- */

    computeForward();

    clearHighlights();

    document
      .querySelectorAll(".edge")
      .forEach(edge => {
        edge.classList.add("forward-active");
      });


    setBadge(
      `Training ${i} / ${numberOfSteps}`,
      "forward"
    );

    $("stepTitle").textContent =
      `Automatic training: ${i} / ${numberOfSteps}`;

    $("stepDescription").textContent =
      "Forward pass → backpropagation → parameter update";

    $("stepCounter").textContent =
      `${i} / ${numberOfSteps}`;

    $("trainingProgress").textContent =
      `${i} / ${numberOfSteps}`;


    $("networkComputationLine1").textContent =
      `Training iteration ${i} of ${numberOfSteps}`;

    $("networkComputationLine2").textContent =
      `Forward: ŷ = ${fmt(values.yHat)}, loss = ${fmt(values.loss, 6)}`;


    updateNetworkValues();
    updateEdgeLabels();
    updateStats();

    await wait(delay);


    /* ----------------------------
       BACKWARD
       ---------------------------- */

    computeBackward();

    clearHighlights();

    document
      .querySelectorAll(".edge")
      .forEach(edge => {
        edge.classList.add("backward-active");
      });


    $("networkComputationLine2").textContent =
      "Backward: gradients recalculated";

    await wait(delay);


    /* ----------------------------
       UPDATE
       ---------------------------- */

    weights.w1 -=
      lr * gradients.dL_dw1;

    weights.w2 -=
      lr * gradients.dL_dw2;

    weights.w3 -=
      lr * gradients.dL_dw3;

    weights.w4 -=
      lr * gradients.dL_dw4;

    weights.w5 -=
      lr * gradients.dL_dw5;

    weights.w6 -=
      lr * gradients.dL_dw6;

    weights.b1 -=
      lr * gradients.dL_db1;

    weights.b2 -=
      lr * gradients.dL_db2;

    weights.b3 -=
      lr * gradients.dL_db3;


    trainingSteps++;


    /*
        Recalculate the network using the newly
        updated parameters.
    */

    computeForward();

    syncBiasInputsFromWeights();


    clearHighlights();

    document
      .querySelectorAll(".edge")
      .forEach(edge => {
        edge.classList.add("updated");
      });


    $("networkComputationLine2").textContent =
      `Update: ŷ = ${fmt(values.yHat)}, loss = ${fmt(values.loss, 6)}`;


    updateNetworkValues();
    updateEdgeLabels();
    updateStats();
    renderWeightTable();


    /*
        Add this completed iteration to history.
    */

    addTrainingHistoryRow(i);


    await wait(delay);
  }


  /*
      ALL requested iterations are now complete.
  */

  computeForward();
  computeBackward();

  syncBiasInputsFromWeights();

  phase = "updated";


  clearHighlights();

  document
    .querySelectorAll(".edge")
    .forEach(edge => {
      edge.classList.add("updated");
    });


  setBadge(
    "Training complete",
    "update"
  );


  $("stepTitle").textContent =
    `Completed ${numberOfSteps} training steps`;

  $("stepDescription").textContent =
    `You completed step 1 manually, then the network automatically repeated the learning cycle ${numberOfSteps - 1} more times.`;

  $("stepCounter").textContent =
    `${numberOfSteps} / ${numberOfSteps}`;

  $("trainingProgress").textContent =
    `${numberOfSteps} / ${numberOfSteps}`;


  $("trainingExplanation").innerHTML =
    `
    <strong>Training complete.</strong>

    You manually worked through the first iteration.

    The network then repeated the same forward pass,
    backpropagation, and parameter update until it completed
    all ${numberOfSteps} training iterations.
    `;


  $("networkComputationLine1").textContent =
    `${numberOfSteps} complete training iterations`;

  $("networkComputationLine2").textContent =
    `Final: ŷ = ${fmt(values.yHat)}, loss = ${fmt(values.loss, 6)}`;


  updateNetworkValues();
  updateEdgeLabels();
  updateStats();
  renderWeightTable();


  resetBtn.disabled = false;

  forwardBtn.disabled = false;
  backwardBtn.disabled = true;
  updateBtn.disabled = true;

  setTrainingButtonsDisabled(false);

  isTraining = false;
}



/* ================================================================
   RESET
   ================================================================ */

function resetDemo() {

  weights =
    { ...INITIAL_WEIGHTS };

  syncBiasInputsFromWeights();

  values = {};

  gradients = {};

  trainingSteps = 0;

  selectedWeight = null;

  showBackpropIdea();
  clearTrainingHistory();

  setActiveTrainingButton(1);

  $("trainingExplanation").textContent =
    "Each training iteration performs a forward pass, calculates the gradients with backpropagation, and then updates the parameters with gradient descent.";

  revealed = {
    h1: false,
    h2: false,
    yHat: false,
    loss: false
};

  phase = "ready";

  forwardStep = 0;
  backwardStep = 0;


  clearHighlights();


  /*
      Display the input immediately but hide calculated values.
  */

  values.x1 =
    Number(x1Input.value);

  values.x2 =
    Number(x2Input.value);

  values.target =
    Number(targetInput.value);


  updateNetworkValues();
  updateEdgeLabels();
  updateStats();
  renderWeightTable();


  setBadge("Ready", "neutral");


  $("stepTitle").textContent =
    "Start with a forward pass";

  $("stepDescription").textContent =
    "The network will compute a prediction from left to right.";

  $("stepCounter").textContent =
    "0 / 0";


  $("calculationTitle").textContent =
    "Waiting to begin";

  $("formula").innerHTML =
    `Press <strong>Step Forward</strong>.`;

  $("calculationDetails").textContent = 
    "";

  $("networkComputationLine1").textContent =
    "Press Step Forward to begin";

  $("networkComputationLine2").textContent =
    "";


  $("upstreamValue").textContent = "—";
  $("localValue").textContent = "—";
  $("wholeValue").textContent = "—";


  $("gradientExplanation").textContent =
    "During the backward pass, each operation receives a gradient from downstream and multiplies it by its own local derivative.";


  $("pennyTitle").textContent =
    "Select a weight";

  $("pennyText").textContent =
    "Once the backward pass has calculated a gradient, click a weight above to ask what would happen if that weight increased by one penny.";

  $("pennyFormula").textContent =
    "ΔL ≈ (∂L/∂w) Δw";

  $("stepTitle").textContent =
    "You completed one full training iteration";

  $("stepDescription").textContent =
    "You manually ran a forward pass, backpropagation, and parameter update. Now try the automatic training controls above to watch this learning cycle repeat."; 

  forwardBtn.disabled = false;
  backwardBtn.disabled = true;
  updateBtn.disabled = true;
}


/* ================================================================
   INPUT CHANGES
   ================================================================ */

function inputChanged() {

  /*
      Changing an input invalidates all previous intermediate
      calculations and gradients.
  */

  phase = "ready";

  gradients = {};

  revealed = {
    h1: false,
    h2: false,
    yHat: false,
    loss: false
};

  forwardStep = 0;
  backwardStep = 0;


  values = {
    x1: Number(x1Input.value),
    x2: Number(x2Input.value),
    target: Number(targetInput.value)
  };


  clearHighlights();
  updateNetworkValues();
  updateStats();
  renderWeightTable();


  setBadge("Ready", "neutral");

  $("stepTitle").textContent =
    "Input changed — run a new forward pass";

  $("stepDescription").textContent =
    "The previous gradients are no longer valid.";

  $("stepCounter").textContent =
    "0 / 0";

  $("networkComputationLine1").textContent =
    "Input changed — run a new forward pass";

  $("networkComputationLine2").textContent =
    "";


  forwardBtn.disabled = false;
  backwardBtn.disabled = true;
  updateBtn.disabled = true;
}


/* ================================================================
   EVENTS
   ================================================================ */

resetBtn.addEventListener(
  "click",
  resetDemo
);


forwardBtn.addEventListener(
  "click",
  stepForward
);


backwardBtn.addEventListener(
  "click",
  stepBackward
);


updateBtn.addEventListener(
  "click",
  updateWeights
);


train1Btn.addEventListener(
  "click",
  () => setActiveTrainingButton(1)
);


train25Btn.addEventListener(
  "click",
  () => setActiveTrainingButton(25)
);


train100Btn.addEventListener(
  "click",
  () => setActiveTrainingButton(100)
);


x1Input.addEventListener(
  "change",
  inputChanged
);


x2Input.addEventListener(
  "change",
  inputChanged
);


targetInput.addEventListener(
  "change",
  inputChanged
);

b1Input.addEventListener(
  "change",
  biasChanged
);

b2Input.addEventListener(
  "change",
  biasChanged
);

b3Input.addEventListener(
  "change",
  biasChanged
);


/* ================================================================
   INITIALIZE
   ================================================================ */

resetDemo();