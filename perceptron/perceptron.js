// ------------------------------------------------------------
// Rosenblatt Perceptron Demo
// Interactive data-space visualization
// ------------------------------------------------------------

const svg = document.getElementById("plot");

const weightSvg =
  document.getElementById("weight-plot");

const NS = "http://www.w3.org/2000/svg";

const WIDTH = 700;
const HEIGHT = 560;

svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);

weightSvg.setAttribute(
  "viewBox",
  `0 0 ${WIDTH} ${HEIGHT}`
);


// ============================================================
// COLORS
// ============================================================

const BLUE = "#2684d8";
const BLUE_SHADE = "#e8eef4";

const ORANGE = "#c97808";

const GREEN = "#087b67";
const GREEN_MARGIN = "#dceee7";

const RED = "#ef4444";
const GRAY = "#6b7280";
const BACKGROUND = "#fffdf9";


// ============================================================
// TRAINING DATA
//
// These are mathematical coordinates, NOT pixel coordinates.
//
// +1 = filled blue
// -1 = hollow
// ============================================================

const data = [
  { x1: -3.6, x2:  2.8, y:  1 },
  { x1: -2.8, x2:  3.5, y:  1 },
  { x1: -2.0, x2:  2.4, y:  1 },
  { x1: -0.8, x2:  3.0, y:  1 },
  { x1:  0.2, x2:  2.0, y:  1 },
  { x1:  1.4, x2:  2.4, y:  1 },
  { x1:  2.4, x2:  1.4, y:  1 },
  { x1:  3.5, x2:  1.0, y:  1 },

  { x1: -3.8, x2:  0.4, y: -1 },
  { x1: -3.0, x2: -0.2, y: -1 },
  { x1: -2.2, x2:  0.3, y: -1 },
  { x1: -1.5, x2: -1.0, y: -1 },
  { x1: -0.5, x2: -0.5, y: -1 },
  { x1:  0.6, x2: -1.4, y: -1 },
  { x1:  1.6, x2: -1.0, y: -1 },
  { x1:  2.6, x2: -2.2, y: -1 },
  { x1:  3.7, x2: -1.8, y: -1 }
];


// ============================================================
// TRUE SEPARATOR
//
// wStar defines the ground-truth separator.
//
// The green dashed line AND green margin are derived from this.
// ============================================================

const wStar = {
  w1: -0.45,
  w2: 1
};

const bStar = 0.35;


// ============================================================
// CURRENT PERCEPTRON
//
// These values will change when Apply Update is clicked.
// ============================================================

const initialW = {
  w1: -0.65,
  w2: 1
};

const initialB = -0.25;

let w = { ...initialW };
let b = initialB;

let currentMistake = null;
let showMistakes = false;


// Which stage of the interaction are we currently in?
//
// "find"   -> Find Mistakes
// "show"   -> Show Update
// "apply"  -> Apply Update

let step = "find";

// Stores the proposed new weight vector during Show Update.
// Importantly, w itself does NOT change until Apply Update.
let proposedW = null;
let proposedB = null;


// ============================================================
// MATHEMATICAL COORDINATE SYSTEM
// ============================================================

const X_MIN = -5;
const X_MAX = 5;

const Y_MIN = -4;
const Y_MAX = 5;


function screenX(x) {

  return (
    (x - X_MIN) /
    (X_MAX - X_MIN)
  ) * WIDTH;
}


function screenY(y) {

  return HEIGHT -
    (
      (y - Y_MIN) /
      (Y_MAX - Y_MIN)
    ) * HEIGHT;
}


// ============================================================
// SVG HELPER
// ============================================================

function addSVG(tag, attributes) {

  const element =
    document.createElementNS(NS, tag);

  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, value);
  }

  svg.appendChild(element);

  return element;
}


// ============================================================
// BACKGROUND
// ============================================================

function drawBackground() {

  addSVG("rect", {
    x: 0,
    y: 0,
    width: WIDTH,
    height: HEIGHT,
    fill: BACKGROUND
  });
}

// ============================================================
// COORDINATE AXES
//
// Show the mathematical x1 and x2 coordinates without
// cluttering the data-space visualization.
// ============================================================

function drawAxes() {

  // We use light labels around the outside of the plot.
  // The graph ranges from:
  //
  // x1: -5 to 5
  // x2: -4 to 5

  const tickColor = "#9ca3af";
  const labelColor = "#6b7280";


  // ----------------------------------------------------------
  // x1 scale along bottom
  // ----------------------------------------------------------

  for (let x = -4; x <= 4; x += 2) {

    const sx = screenX(x);

    // small tick
    addSVG("line", {
      x1: sx,
      y1: HEIGHT - 6,
      x2: sx,
      y2: HEIGHT,
      stroke: tickColor,
      "stroke-width": 1
    });

    // number
    const text = addSVG("text", {
      x: sx,
      y: HEIGHT - 12,
      "text-anchor": "middle",
      fill: labelColor,
      "font-size": 12
    });

    text.textContent = x;
  }


  // x1 label
  const xLabel = addSVG("text", {
    x: WIDTH - 12,
    y: HEIGHT - 12,
    "text-anchor": "end",
    fill: "#444",
    "font-size": 17,
    "font-style": "italic"
  });

  xLabel.textContent = "x₁";


  // ----------------------------------------------------------
  // x2 scale along left
  // ----------------------------------------------------------

  for (let y = -2; y <= 4; y += 2) {

    const sy = screenY(y);

    // small tick
    addSVG("line", {
      x1: 0,
      y1: sy,
      x2: 6,
      y2: sy,
      stroke: tickColor,
      "stroke-width": 1
    });

    // number
    const text = addSVG("text", {
      x: 12,
      y: sy - 6,
      fill: labelColor,
      "font-size": 12
    });

    text.textContent = y;
  }


  // x2 label
  const yLabel = addSVG("text", {
    x: 12,
    y: 22,
    fill: "#444",
    "font-size": 17,
    "font-style": "italic"
  });

  yLabel.textContent = "x₂";
}

// ============================================================
// WEIGHT SPACE
//
// Data space uses coordinates (x1, x2).
//
// Weight space uses coordinates (w1, w2).
// Every possible weight vector corresponds to a location
// in this graph.
// ============================================================

const W_MIN = -4;
const W_MAX = 4;


// Convert weight coordinates into SVG coordinates.

function weightScreenX(w1) {

  return (
    (w1 - W_MIN) /
    (W_MAX - W_MIN)
  ) * WIDTH;
}


function weightScreenY(w2) {

  return HEIGHT -
    (
      (w2 - W_MIN) /
      (W_MAX - W_MIN)
    ) * HEIGHT;
}


// Add an SVG element to the RIGHT graph.

function addWeightSVG(tag, attributes) {

  const element =
    document.createElementNS(NS, tag);

  for (
    const [name, value]
    of Object.entries(attributes)
  ) {

    element.setAttribute(name, value);
  }

  weightSvg.appendChild(element);

  return element;
}


// ============================================================
// WEIGHT-SPACE ARROWHEAD
// ============================================================

function makeWeightArrowMarker(
  id,
  color
) {

  const defs =
    document.createElementNS(NS, "defs");

  const marker =
    document.createElementNS(NS, "marker");

  marker.setAttribute("id", id);
  marker.setAttribute("markerWidth", "10");
  marker.setAttribute("markerHeight", "10");
  marker.setAttribute("refX", "8");
  marker.setAttribute("refY", "3");
  marker.setAttribute("orient", "auto");


  const arrow =
    document.createElementNS(NS, "path");

  arrow.setAttribute(
    "d",
    "M0,0 L0,6 L9,3 z"
  );

  arrow.setAttribute(
    "fill",
    color
  );


  marker.appendChild(arrow);
  defs.appendChild(marker);
  weightSvg.appendChild(defs);
}


// ============================================================
// DRAW A VECTOR IN WEIGHT SPACE
// ============================================================

function drawWeightSpaceVector(
  weights,
  color,
  markerId,
  label,
  dashed = false
) {

  const magnitude =
    Math.hypot(
      weights.w1,
      weights.w2
    );


  if (magnitude === 0) {
    return;
  }


  // Fixed visual length.
  //
  // For now we care primarily about seeing the direction
  // of each weight vector clearly.

  const displayLength = 2.7;


  const endW1 =
    (weights.w1 / magnitude) *
    displayLength;

  const endW2 =
    (weights.w2 / magnitude) *
    displayLength;


  const attributes = {

    x1: weightScreenX(0),
    y1: weightScreenY(0),

    x2: weightScreenX(endW1),
    y2: weightScreenY(endW2),

    stroke: color,

    "stroke-width": 4,

    "marker-end":
      `url(#${markerId})`
  };


  if (dashed) {

    attributes["stroke-dasharray"] =
      "9 7";
  }


  addWeightSVG(
    "line",
    attributes
  );


  const labelText =
    addWeightSVG("text", {

      x:
        weightScreenX(endW1) + 10,

      y:
        weightScreenY(endW2) - 8,

      fill: color,

      "font-size": 18,

      "font-style": "italic"
    });


  labelText.textContent = label;
}


// ============================================================
// DRAW THE ENTIRE WEIGHT-SPACE PANEL
// ============================================================

function drawWeightSpace() {

  // Clear only the right graph.

  weightSvg.innerHTML = "";


  // ----------------------------------------------------------
  // Background
  // ----------------------------------------------------------

  addWeightSVG("rect", {

    x: 0,
    y: 0,

    width: WIDTH,
    height: HEIGHT,

    fill: BACKGROUND
  });


  // ----------------------------------------------------------
  // Axes
  // ----------------------------------------------------------

  // Horizontal w1 axis

  addWeightSVG("line", {

    x1: weightScreenX(W_MIN),
    y1: weightScreenY(0),

    x2: weightScreenX(W_MAX),
    y2: weightScreenY(0),

    stroke: "#d1d5db",
    "stroke-width": 1.5
  });


  // Vertical w2 axis

  addWeightSVG("line", {

    x1: weightScreenX(0),
    y1: weightScreenY(W_MIN),

    x2: weightScreenX(0),
    y2: weightScreenY(W_MAX),

    stroke: "#d1d5db",
    "stroke-width": 1.5
  });


  // Origin

  addWeightSVG("circle", {

    cx: weightScreenX(0),
    cy: weightScreenY(0),

    r: 5,

    fill: "#374151"
  });


  // ----------------------------------------------------------
  // Axis labels
  // ----------------------------------------------------------

  const w1Label =
    addWeightSVG("text", {

      x: WIDTH - 18,
      y: weightScreenY(0) - 10,

      fill: "#555",

      "font-size": 17,
      "font-style": "italic"
    });

  w1Label.textContent = "w₁";


  const w2Label =
    addWeightSVG("text", {

      x: weightScreenX(0) + 10,
      y: 22,

      fill: "#555",

      "font-size": 17,
      "font-style": "italic"
    });

  w2Label.textContent = "w₂";


  // ----------------------------------------------------------
  // Arrowheads
  // ----------------------------------------------------------

  makeWeightArrowMarker(
    "weight-current-arrow",
    ORANGE
  );

  makeWeightArrowMarker(
    "weight-proposed-arrow",
    "#7c3aed"
  );

  makeWeightArrowMarker(
    "weight-star-arrow",
    GREEN
  );


  // ----------------------------------------------------------
  // TRUE / TARGET DIRECTION w*
  // ----------------------------------------------------------

  drawWeightSpaceVector(
    wStar,
    GREEN,
    "weight-star-arrow",
    "w*",
    false
  );


  // ----------------------------------------------------------
  // CURRENT w
  // ----------------------------------------------------------

  drawWeightSpaceVector(
    w,
    ORANGE,
    "weight-current-arrow",
    "w"
  );


  // ----------------------------------------------------------
  // PROPOSED w_new
  //
  // This appears only after Show Update.
  // ----------------------------------------------------------

  if (
    step === "apply" &&
    proposedW
  ) {

    drawWeightSpaceVector(
      proposedW,
      "#7c3aed",
      "weight-proposed-arrow",
      "w_new"
    );
  }
}


// ============================================================
// LINE FROM w AND b
//
// Decision boundary:
//
//      w1*x1 + w2*x2 + b = 0
//
// Solve for x2:
//
//      x2 = -(w1*x1 + b) / w2
// ============================================================

function boundaryY(weights, bias, x) {

  return -(
    weights.w1 * x + bias
  ) / weights.w2;
}


// ============================================================
// CURRENT CLASSIFICATION REGION
//
// Shade the side where the current perceptron predicts -1.
// ============================================================

function drawCurrentRegion() {

  const leftY =
    boundaryY(w, b, X_MIN);

  const rightY =
    boundaryY(w, b, X_MAX);


  addSVG("polygon", {

    points: `
      ${screenX(X_MIN)},${screenY(leftY)}
      ${screenX(X_MAX)},${screenY(rightY)}
      ${screenX(X_MAX)},${screenY(Y_MIN)}
      ${screenX(X_MIN)},${screenY(Y_MIN)}
    `,

    fill: BLUE_SHADE
  });
}


// ============================================================
// TRUE MARGIN
//
// Find the closest training point to the true separator.
//
// For a separator:
//
//      w*ᵀx + b* = 0
//
// the perpendicular distance of a point from the line is:
//
//      |w*ᵀx + b*| / ||w*||
//
// The closest training point determines the margin.
//
// and fill the space between them.
// ============================================================

function drawTrueMargin() {

  // Size of the margin around the true separator.
  // No training points should lie inside this band.
  const margin = 0.45;


  // Upper edge of margin
  const upperLeft =
    -(wStar.w1 * X_MIN + bStar - margin) /
    wStar.w2;

  const upperRight =
    -(wStar.w1 * X_MAX + bStar - margin) /
    wStar.w2;


  // Lower edge of margin
  const lowerLeft =
    -(wStar.w1 * X_MIN + bStar + margin) /
    wStar.w2;

  const lowerRight =
    -(wStar.w1 * X_MAX + bStar + margin) /
    wStar.w2;


  // Draw the green margin band
  addSVG("polygon", {

    points: `
      ${screenX(X_MIN)},${screenY(upperLeft)}
      ${screenX(X_MAX)},${screenY(upperRight)}
      ${screenX(X_MAX)},${screenY(lowerRight)}
      ${screenX(X_MIN)},${screenY(lowerLeft)}
    `,

    fill: GREEN_MARGIN
  });
}


// ============================================================
// TRUE SEPARATOR
//
// Automatically centered inside the margin.
// ============================================================

function drawTrueSeparator() {

  const leftY =
    boundaryY(wStar, bStar, X_MIN);

  const rightY =
    boundaryY(wStar, bStar, X_MAX);


  addSVG("line", {

    x1: screenX(X_MIN),
    y1: screenY(leftY),

    x2: screenX(X_MAX),
    y2: screenY(rightY),

    stroke: GREEN,
    "stroke-width": 2.5,
    "stroke-dasharray": "10 9"
  });
}


// ============================================================
// CURRENT DECISION BOUNDARY
//
// Derived directly from current w and b.
// ============================================================

function drawCurrentBoundary() {

  const leftY =
    boundaryY(w, b, X_MIN);

  const rightY =
    boundaryY(w, b, X_MAX);


  addSVG("line", {

    x1: screenX(X_MIN),
    y1: screenY(leftY),

    x2: screenX(X_MAX),
    y2: screenY(rightY),

    stroke: ORANGE,
    "stroke-width": 4
  });
}


// ============================================================
// CURRENT WEIGHT VECTOR
//
// IMPORTANT:
//
// The decision boundary has normal vector w.
//
// Therefore we don't calculate some separate perpendicular
// slope. We literally draw the vector w itself.
//
// This guarantees that the arrow stays perpendicular to
// the decision boundary as w changes.
// ============================================================

function drawWeightVector() {

  // ==========================================================
  // CURRENT WEIGHT VECTOR w
  //
  // w is the normal vector of the current decision boundary,
  // so it is perpendicular to the orange boundary.
  // ==========================================================

  // Anchor w on the current decision boundary.
  const startX = 0;
  const startY = boundaryY(w, b, startX);


  // Fixed visual length.
  // We use the direction of w, not its numerical magnitude,
  // to determine how the arrow appears.
  const displayLength = 2.2;

  const magnitude =
    Math.hypot(w.w1, w.w2);

  const endX =
    startX +
    (w.w1 / magnitude) * displayLength;

  const endY =
    startY +
    (w.w2 / magnitude) * displayLength;


  // ==========================================================
  // ARROWHEAD
  // ==========================================================

  const defs =
    document.createElementNS(NS, "defs");

  const marker =
    document.createElementNS(NS, "marker");

  marker.setAttribute("id", "w-arrow");
  marker.setAttribute("markerWidth", "10");
  marker.setAttribute("markerHeight", "10");
  marker.setAttribute("refX", "8");
  marker.setAttribute("refY", "3");
  marker.setAttribute("orient", "auto");


  const arrow =
    document.createElementNS(NS, "path");

  arrow.setAttribute(
    "d",
    "M0,0 L0,6 L9,3 z"
  );

  arrow.setAttribute(
    "fill",
    ORANGE
  );


  marker.appendChild(arrow);
  defs.appendChild(marker);
  svg.appendChild(defs);


  // ==========================================================
  // DRAW w
  // ==========================================================

  addSVG("line", {

    x1: screenX(startX),
    y1: screenY(startY),

    x2: screenX(endX),
    y2: screenY(endY),

    stroke: ORANGE,
    "stroke-width": 4,

    "marker-end": "url(#w-arrow)"
  });


  // ==========================================================
  // LABEL
  // ==========================================================

  const label =
    addSVG("text", {

      x: screenX(endX) + 8,
      y: screenY(endY) + 3,

      fill: "#555",

      "font-size": 18,
      "font-style": "italic"
    });

  label.textContent = "w";
}

// ============================================================
// SHOW THE PROPOSED PERCEPTRON UPDATE
//
// Visualizes:
//
//        w_old + yx = w_new
//
// without actually changing w yet.
// ============================================================

function drawProposedUpdate() {

  if (!currentMistake || !proposedW) {
    return;
  }

  const y = currentMistake.y;

  // ==========================================================
  // CORRECTION VECTOR: yx
  // ==========================================================

  const correction = {
    w1: y * currentMistake.x1,
    w2: y * currentMistake.x2
  };


  // ==========================================================
  // ARROWHEADS
  // ==========================================================

  const defs =
    document.createElementNS(NS, "defs");


  function makeMarker(id, color) {

    const marker =
      document.createElementNS(NS, "marker");

    marker.setAttribute("id", id);
    marker.setAttribute("markerWidth", "10");
    marker.setAttribute("markerHeight", "10");
    marker.setAttribute("refX", "8");
    marker.setAttribute("refY", "3");
    marker.setAttribute("orient", "auto");


    const arrow =
      document.createElementNS(NS, "path");

    arrow.setAttribute(
      "d",
      "M0,0 L0,6 L9,3 z"
    );

    arrow.setAttribute(
      "fill",
      color
    );


    marker.appendChild(arrow);
    defs.appendChild(marker);
  }


  makeMarker(
    "yx-arrow",
    "#dc2626"
  );

  makeMarker(
    "new-w-arrow",
    "#7c3aed"
  );

  svg.appendChild(defs);


  // ==========================================================
  // RED yx
  //
  // Draw from the common vector origin on the CURRENT
  // decision boundary TO the selected mistake.
  // ==========================================================

  // Same origin used by w and w_new
  const vectorOriginX = 0;

  const vectorOriginY =
    boundaryY(
        w,
        b,
        vectorOriginX
    );

  // The selected misclassified point
  const mistakeX = currentMistake.x1;
  const mistakeY = currentMistake.x2;


  // Draw arrow FROM decision boundary TO mistake
  addSVG("line", {

    x1: screenX(vectorOriginX),
    y1: screenY(vectorOriginY),

    x2: screenX(mistakeX),
    y2: screenY(mistakeY),

    stroke: "#dc2626",
    "stroke-width": 3,

    "marker-end": "url(#yx-arrow)"
    });


  // y·x label near the mistake
  const correctionLabel =
    addSVG("text", {

        x: screenX(mistakeX) + 10,
        y: screenY(mistakeY) - 10,

        fill: "#dc2626",

        "font-size": 17,
        "font-style": "italic"
    });

  correctionLabel.textContent = "y·x";


  // ==========================================================
  // PURPLE w_new PREVIEW
  //
  // Purple begins at the SAME place as the current orange w.
  //
  // It shows the DIRECTION that w will have after the update.
  //
  // After Apply Update, the new orange w will have this exact
  // direction, but its tail will be translated onto the new
  // decision boundary.
  // ==========================================================

  const proposedStartX = 0;

  const proposedStartY =
    boundaryY(
      w,
      b,
      proposedStartX
    );


  const proposedMagnitude =
    Math.hypot(
      proposedW.w1,
      proposedW.w2
    );


  const displayLength = 2.2;


  const proposedEndX =
    proposedStartX +
    (proposedW.w1 / proposedMagnitude) *
    displayLength;

  const proposedEndY =
    proposedStartY +
    (proposedW.w2 / proposedMagnitude) *
    displayLength;


  addSVG("line", {

    x1:
      screenX(proposedStartX),

    y1:
      screenY(proposedStartY),

    x2:
      screenX(proposedEndX),

    y2:
      screenY(proposedEndY),

    stroke:
      "#7c3aed",

    "stroke-width":
      3,

    "marker-end":
      "url(#new-w-arrow)"
  });


  // w_new label

  const newLabel =
    addSVG("text", {

      x:
        screenX(proposedEndX) + 8,

      y:
        screenY(proposedEndY) + 3,

      fill:
        "#7c3aed",

      "font-size":
        16,

      "font-style":
        "italic"
    });

  newLabel.textContent = "w_new";
}
// ============================================================
// POINTS
// ============================================================

function drawPoints() {

  data.forEach(point => {

    const cx =
      screenX(point.x1);

    const cy =
      screenY(point.x2);


    // RED RING ONLY FOR THE SELECTED MISTAKE.
    //
    // Nothing is red when the demo first loads.

    // Calculate the perceptron's prediction for this point
const score =
  w.w1 * point.x1 +
  w.w2 * point.x2 +
  b;

// Rosenblatt mistake condition
const isMistake =
  point.y * score <= 0;


// Only reveal mistakes after Find Mistake is clicked
if (showMistakes && isMistake) {

  const isSelected =
    point === currentMistake;

  addSVG("circle", {

    cx,
    cy,

    r: isSelected ? 16 : 14,

    fill: "none",

    stroke: RED,

    "stroke-width":
      isSelected ? 4 : 2
  });
}

    // True +1
    if (point.y === 1) {

      addSVG("circle", {

        cx,
        cy,

        r: 8,

        fill: BLUE,
        stroke: BLUE,
        "stroke-width": 2
      });
    }


    // True -1
    else {

      addSVG("circle", {

        cx,
        cy,

        r: 8,

        fill: BACKGROUND,

        stroke: GRAY,
        "stroke-width": 2
      });
    }
  });
}


// ============================================================
// REDRAW
// ============================================================

function redraw() {

  svg.innerHTML = "";

  drawBackground();

  drawCurrentRegion();

  drawTrueMargin();

  drawTrueSeparator();

  drawCurrentBoundary();

  drawWeightVector();

  if (step === "apply") {
    drawProposedUpdate();
  }

  drawPoints();

  drawAxes();

  drawWeightSpace();
}


// ============================================================
// FIND MISTAKE
// ============================================================

function findMistake() {

  showMistakes = true;


  const mistakes =
    data.filter(point => {

      const score =
        w.w1 * point.x1 +
        w.w2 * point.x2 +
        b;

      return point.y * score <= 0;
    });


  currentMistake =
    mistakes.length > 0
      ? mistakes[0]
      : null;


  if (currentMistake) {

    const score =
      w.w1 * currentMistake.x1 +
      w.w2 * currentMistake.x2 +
      b;


    const prediction =
      score >= 0 ? "+1" : "-1";


    step = "show";

    document.getElementById(
      "next-step"
    ).textContent = "Show Update";


    document.getElementById(
      "explanation"
    ).innerHTML = `

      <strong>
        ${mistakes.length} misclassified
        ${mistakes.length === 1 ? "point" : "points"} found
      </strong>

      <p>
        Red rings show every point the current perceptron
        classifies incorrectly.
      </p>

      <p>
        The perceptron processes one mistake at a time.
        The thicker red ring marks the first mistake in the
        current training order.
      </p>

      <p>
        <strong>Selected:</strong>
        x = (${currentMistake.x1}, ${currentMistake.x2}),
        true label y =
        ${currentMistake.y > 0 ? "+1" : "-1"}
      </p>

      <p>
        The perceptron calculates a classification score:
      </p>

      <p style="font-size: 1.1rem;">
        <strong>
          score = wᵀx + b
        </strong>
      </p>

      <p>
        <strong>w</strong> is the current weight vector,
        <strong>x</strong> is the input point, and
        <strong>b</strong> is the bias.
        A positive score predicts +1; a negative score predicts −1.
      </p>

      <p>
        Here,
        <strong>wᵀx + b = ${score.toFixed(2)}</strong>,
        so the perceptron predicts ${prediction},
        but the true label is
        ${currentMistake.y > 0 ? "+1" : "-1"}.
      </p>

      <p>
        Click <strong>Show Update</strong> to see how this
        mistake will change the weight vector.
      </p>
    `;
  }

  else {

    step = "find";

    document.getElementById(
      "next-step"
    ).textContent = "Find Mistakes";


    document.getElementById(
      "explanation"
    ).innerHTML = `

      <strong>No mistakes remain.</strong>

      <p>
        Every training point is currently classified correctly.
      </p>
    `;
  }


  redraw();
}

function showUpdate() {

  if (!currentMistake) {
    return;
  }


  const y =
    currentMistake.y;


  // Calculate what w WOULD become.
  //
  // Do not change w yet.

  proposedW = {

    w1:
      w.w1 +
      y * currentMistake.x1,

    w2:
      w.w2 +
      y * currentMistake.x2
  };


  proposedB =
    b + y;


  step = "apply";


  document.getElementById(
    "next-step"
  ).textContent = "Apply Update";


  document.getElementById(
    "explanation"
  ).innerHTML = `

    <strong>
      Proposed perceptron update
    </strong>

    <p>
      The selected point was misclassified, so the perceptron
      uses that point and its true label to correct the weight
      vector.
    </p>


    <p style="font-size: 1.25rem;">
      <strong>
        w<sub>new</sub> ← w + yx
      </strong>
    </p>


    <p>
      <strong>w</strong> = current weight vector<br>
      <strong>x</strong> = selected misclassified point<br>
      <strong>y</strong> = its true label (+1 or −1)<br>
      <strong>yx</strong> = the correction vector
    </p>


    <p style="font-size: 1.15rem;">
      <strong>
        w<sub>old</sub>
        &nbsp;+&nbsp;
        yx
        &nbsp;=&nbsp;
        w<sub>new</sub>
      </strong>
    </p>

    <p>
      <em>
        where we are
        &nbsp;+&nbsp;
        correction
        &nbsp;=&nbsp;
        where we move
      </em>
    </p>


    <p>
      For this mistake:
    </p>

    <p>
      w<sub>old</sub>
      =
      (${w.w1.toFixed(2)}, ${w.w2.toFixed(2)})
    </p>

    <p>
      yx
      =
      (${y})(${currentMistake.x1}, ${currentMistake.x2})
      =
      (${(y * currentMistake.x1).toFixed(2)},
       ${(y * currentMistake.x2).toFixed(2)})
    </p>

    <p>
      Therefore:
    </p>

    <p style="font-size: 1.15rem;">
      <strong>
        w<sub>new</sub>
        =
        (${proposedW.w1.toFixed(2)},
         ${proposedW.w2.toFixed(2)})
      </strong>
    </p>


    <p>
        <strong> On the graph (left):</strong>
        <strong style="color:#c97808;">w</strong>
        is the current weight vector.
        The
        <strong style="color:#dc2626;">yx</strong>
        arrow shows the correction contributed by the selected
        misclassified point.
        The
        <strong style="color:#7c3aed;">w<sub>new</sub></strong>
        arrow previews where the weight vector will point after
        the update.
    </p>

    <p>
        <strong>In Weight Space (right):</strong>
        the update is shown directly in terms of the model's weights.
        The <strong style="color:#c97808;">orange w</strong>
        is the current weight vector, and the
        <strong style="color:#7c3aed;">purple w<sub>new</sub></strong>
        shows the proposed new weight direction after applying this
        correction.
    </p>

    <p>
        The <strong style="color:#087b67;">teal w*</strong>
        is the target weight direction associated with the true separator
        in Data Space. It is shown as a reference only—the perceptron
        does not know <em>w*</em> or directly try to move toward it.
    </p>

    <p>
        The update is:
    </p>

    <p style="font-size: 1.2rem;">
        <strong>
            w<sub>new</sub> ← w + yx
        </strong>
    </p>

    <p>
        <strong>w</strong> = where the weight vector is now<br>
        <strong>yx</strong> = the correction supplied by this mistake<br>
        <strong>w<sub>new</sub></strong> = where the weight vector
        will be after applying that correction
    </p>

    <p>
        Click <strong>Apply Update</strong>.
        The purple preview will disappear and the orange
        weight vector will move to that new direction.
    </p>

    `;

  redraw();
}

function applyUpdate() {

  if (!currentMistake || !proposedW) {
    return;
  }


  // Save values for explanation
  const oldW = { ...w };
  const oldB = b;

  const usedPoint = { ...currentMistake };
  const y = currentMistake.y;


  // ----------------------------------------------------------
  // ACTUALLY APPLY THE UPDATE
  // ----------------------------------------------------------

  w = { ...proposedW };
  b = proposedB;


  // Clear the proposed update
  proposedW = null;
  proposedB = null;

  currentMistake = null;

  // Hide mistakes again.
  // The user must click Find Mistakes to inspect the new model.
  showMistakes = false;


  // Return to first interaction stage
  step = "find";


  document.getElementById(
    "next-step"
  ).textContent = "Find Mistakes";


  document.getElementById(
    "explanation"
  ).innerHTML = `

    <strong>
      Update applied
    </strong>

    <p>
      The weight vector has moved from
      <strong>
        (${oldW.w1.toFixed(2)}, ${oldW.w2.toFixed(2)})
      </strong>
      to
      <strong>
        (${w.w1.toFixed(2)}, ${w.w2.toFixed(2)})
      </strong>.
    </p>

    <p>
      The bias also changed:
      <strong>
        ${oldB.toFixed(2)} → ${b.toFixed(2)}
      </strong>.
    </p>

    <p>
      Because the orange decision boundary is defined by
      <strong>wᵀx + b = 0</strong>,
      changing w and b changes the classifier itself.
    </p>

   <p>
        Notice how the same update appears in both panels.
        In <strong>Weight Space</strong>, the orange
        <strong>w</strong> has moved to its new direction.
        In <strong>Data Space</strong>, changing
        <strong>w</strong> and <strong>b</strong> changes the orientation
        and position of the orange decision boundary.
    </p>

    <p>
      Click <strong>Find Mistakes</strong> to test the updated
      classifier.
    </p>
  `;


  redraw();
}


// ============================================================
// RESTART
// ============================================================

function restart() {

  w = { ...initialW };

  b = initialB;

  currentMistake = null;
  showMistakes = false;

  proposedW = null;
  proposedB = null;

  step = "find";

  document.getElementById(
    "next-step"
  ).textContent = "Find Mistakes";


  document.getElementById(
    "explanation"
  ).innerHTML = `

    <p>
        <strong>Two views of the perceptron:</strong>
        The panels show the same classifier from two different perspectives.
        <strong>Data Space</strong> shows the training examples and the
        decision boundary they are being classified by.
        <strong>Weight Space</strong> shows the model itself in terms of its
        weights <em>(w₁, w₂)</em>.
    </p>

    <p>
        In Weight Space, the
        <strong style="color:#c97808;">orange w</strong>
        is the perceptron's current weight vector. The
        <strong style="color:#087b67;">teal w*</strong>
        is the target weight direction associated with the true separator
        shown by the dashed teal line in Data Space.
        The perceptron does not know <em>w*</em>; it is shown only as a
        reference so we can watch how learning changes <em>w</em>.
    </p>

    <p>
      Start by finding a point that the current
      perceptron misclassifies.
    </p>
  `;


  redraw();
}


// ============================================================
// BUTTONS
// ============================================================

document
  .getElementById("next-step")
  .addEventListener("click", () => {

    if (step === "find") {
      findMistake();
    }

    else if (step === "show") {
      showUpdate();
    }

    else if (step === "apply") {
      applyUpdate();
    }

  });


document
  .getElementById("restart")
  .addEventListener("click", restart);

document
  .getElementById("restart")
  .addEventListener("click", restart);


// Initial render
redraw();