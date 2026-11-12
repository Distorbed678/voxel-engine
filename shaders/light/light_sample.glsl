#ifndef LIGHT_SAMPLE_GLSL
#define LIGHT_SAMPLE_GLSL
// Smooth lighting for surfaces: trilinear sample of the light volume `offset` blocks from the
// surface point (half a block out along the normal for faces). rgb = block light, a = sky light.
// Outside the volume (and fading in over its outer 12 blocks): no block light, open sky.
#include "common.glsl"
#include "light/light_defs.glsl"

// block light colour scale (light values are squared first for a soft falloff)
#define BLOCK_LIGHT_GAIN 2.4
// self-illumination of emissive faces
#define EMISSIVE_GAIN 5.0
// what's left in a sealed cave with no light at all
#define CAVE_AMBIENT vec3(0.018)

vec4 sample_light(FrameData f, vec3 rel, vec3 offset) {
    const vec4 OUTSIDE = vec4(0.0, 0.0, 0.0, 1.0);
    if ((f.lightOrigin.w & LIGHT_ENABLED) == 0) return OUTSIDE;
    vec3 p = rel + f.camFrac.xyz + offset;                // relative to the camera block
    vec3 local = p - vec3(f.lightOrigin.xyz);
    vec3 edge = min(local, vec3(float(LIGHT_N)) - local);
    float fade = smoothstep(1.0, 12.0, min(edge.x, min(edge.y, edge.z)));
    if (fade <= 0.0) return OUTSIDE;
    vec3 tp = vec3(f.camBlock.xyz & LIGHT_MASK) + p;      // texel space, wraps (repeat sampler)
    vec4 L = textureLod(gTex3D[nonuniformEXT(f.lightTex)], tp / float(LIGHT_N), 0.0);
    return mix(OUTSIDE, L, fade);
}

#endif
