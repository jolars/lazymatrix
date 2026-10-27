<script setup>
defineProps({
  plot: { type: Object, required: true },
  title: { type: String, required: true },
  alt: { type: String, required: true },
});
</script>

<template>
  <figure class="benchmark-figure">
    <h3>{{ title }}</h3>
    <picture>
      <source
        media="(max-width: 639px)"
        :srcset="plot.narrow"
        width="320"
        height="340"
      />
      <img :src="plot.wide" width="520" height="340" :alt="alt" />
    </picture>
    <figcaption>
      <slot />
      <a :href="plot.wide" download>Download SVG</a>.
    </figcaption>
  </figure>
</template>

<style scoped>
figure {
  min-width: 0;
  margin: 0;
}
.benchmark-figure h3 {
  margin: 0 0 0.75rem;
  font-size: 1rem;
}
picture {
  display: block;
  background: white;
}
img {
  display: block;
  width: 100%;
  height: auto;
}
figcaption {
  margin-top: 0.75rem;
  color: var(--vp-c-text-2);
  font-size: 0.85rem;
  line-height: 1.6;
}
</style>
