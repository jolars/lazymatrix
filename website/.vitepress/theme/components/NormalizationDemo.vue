<script setup>
import katex from "katex";

const math = (tex) => katex.renderToString(tex, { throwOnError: true });
const leftParenthesis = math(
  String.raw`\left(\vphantom{\begin{bmatrix}3&6&0\\0&6&0\\0&0&9\end{bmatrix}}\right.`,
);
const rightParenthesis = math(
  String.raw`\left.\vphantom{\begin{bmatrix}3&6&0\\0&6&0\\0&0&9\end{bmatrix}}\right)`,
);
const terms = [
  {
    symbol: "X",
    matrix: String.raw`\begin{bmatrix}3&6&\phantom{0}\\\phantom{0}&6&\phantom{0}\\\phantom{0}&\phantom{0}&9\end{bmatrix}`,
    label: "Original matrix",
    detail: "Only four stored values",
  },
  {
    symbol: String.raw`C = \mathbf{1}c^\top`,
    matrix: String.raw`\begin{bmatrix}1&4&3\\1&4&3\\1&4&3\end{bmatrix}`,
    label: "Column centers, repeated",
    detail: "Represented by c = [1, 4, 3]",
  },
  {
    symbol: String.raw`S^{-1}`,
    matrix: String.raw`\begin{bmatrix}\frac12&\phantom{0}&\phantom{0}\\\phantom{0}&\frac14&\phantom{0}\\\phantom{0}&\phantom{0}&\frac16\end{bmatrix}`,
    label: "Inverse column scales",
    detail: "Represented by s = [2, 4, 6]",
  },
];
const expanded = math(
  String.raw`\widetilde{X} = \begin{bmatrix}
    \dfrac{3-1}{2} & \dfrac{6-4}{4} & \dfrac{0-3}{6}\\[1em]
    \dfrac{0-1}{2} & \dfrac{6-4}{4} & \dfrac{0-3}{6}\\[1em]
    \dfrac{0-1}{2} & \dfrac{0-4}{4} & \dfrac{9-3}{6}
  \end{bmatrix} = \begin{bmatrix}
    1 & \dfrac12 & \color{#3b82f6}{-\dfrac12}\\[1em]
    \color{#3b82f6}{-\dfrac12} & \dfrac12 & \color{#3b82f6}{-\dfrac12}\\[1em]
    \color{#3b82f6}{-\dfrac12} & \color{#3b82f6}{-1} & 1
  \end{bmatrix}`,
);
</script>

<template>
  <figure
    class="normalization-demo"
    aria-label="Expanded normalization equation"
  >
    <div
      class="equation-scroll"
      tabindex="0"
      role="region"
      aria-label="The three matrices in (X minus C) times inverse S"
    >
      <div class="matrix-equation">
        <span
          class="parenthesis"
          style="grid-column: 1"
          aria-hidden="true"
          v-html="leftParenthesis"
        ></span>
        <template v-for="(term, i) in terms" :key="term.symbol">
          <span
            v-if="i === 1"
            class="operator"
            style="grid-column: 3"
            aria-label="minus"
            v-html="math('-')"
          ></span>
          <span
            v-if="i === 2"
            class="parenthesis"
            style="grid-column: 5"
            aria-hidden="true"
            v-html="rightParenthesis"
          ></span>
          <span
            v-if="i === 2"
            class="operator"
            style="grid-column: 6"
            aria-label="times"
            v-html="math(String.raw`\times`)"
          ></span>
          <div class="matrix-term" :style="{ '--term-column': [2, 4, 7][i] }">
            <div class="symbol" v-html="math(term.symbol)"></div>
            <div class="matrix" v-html="math(term.matrix)"></div>
            <div class="term-label">{{ term.label }}</div>
            <div class="term-detail">{{ term.detail }}</div>
          </div>
        </template>
      </div>
    </div>

    <div class="expanded-result">
      <p>
        Without LazyMatrix, you either have to forget about normalizing the
        matrix, or realize the normalized matrix as a dense one:
      </p>
      <div
        class="equation-scroll result-math"
        tabindex="0"
        role="region"
        aria-label="Entry by entry normalization and its dense result"
        v-html="expanded"
      ></div>
      <p>
        The <span class="highlight-color">blue</span> values were implicit
        zeros. LazyMatrix keeps <strong>X</strong> and the two short vectors
        <strong>c</strong> and <strong>s</strong>. The repeated centers,
        diagonal matrix, and normalized entries above describe the operation;
        they never need to be stored as matrices.
      </p>
    </div>
  </figure>
</template>

<style scoped>
.highlight-color {
  color: #3b82f6;
}
.normalization-demo {
  margin: 16px 0;
}
.equation-scroll {
  overflow-x: auto;
  padding: 8px 0;
}
.equation-scroll:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: -3px;
}
.matrix-equation {
  display: grid;
  grid-template-columns: repeat(7, max-content);
  grid-template-rows: auto 120px auto auto;
  align-items: center;
  justify-content: center;
  column-gap: 12px;
  min-width: 640px;
  padding: 16px 0;
}
.matrix-term {
  display: contents;
}
.matrix-term > * {
  grid-column: var(--term-column);
  text-align: center;
}
.symbol {
  grid-row: 1;
  color: var(--vp-c-brand-1);
  padding-bottom: 16px;
}
.matrix {
  grid-row: 2;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
}
.matrix-term:last-child .term-label,
.matrix-term:last-child .term-detail {
  width: 0;
  min-width: 100%;
  display: flex;
  justify-content: center;
  white-space: nowrap;
  text-align: center;
}
.operator,
.parenthesis {
  grid-row: 2;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  color: var(--vp-c-text-1);
}
.parenthesis {
  font-size: 21px;
  color: var(--vp-c-text-2);
}
.parenthesis:first-child {
  transform: translateX(12px);
}
.parenthesis:not(:first-child) {
  margin-left: -12px;
}
.term-label {
  grid-row: 3;
  padding-top: 16px;
  font-size: 13px;
  font-weight: 600;
}
.term-detail {
  grid-row: 4;
  font-size: 12px;
  color: var(--vp-c-text-2);
}
.expanded-result {
  margin-top: 16px;
}
.result-math {
  text-align: center;
  padding: 20px 0;
}
figcaption {
  margin-top: 16px;
  line-height: 1.75;
}
.normalization-demo .scale-note {
  margin: 8px 0 16px;
  font-size: 12px;
  color: var(--vp-c-text-2);
}
</style>
