#ifndef LIGHT_SAMPLE_GLSL
#define LIGHT_SAMPLE_GLSL
// Smooth lighting for surfaces: trilinear sample of the light volume `offset` blocks from the
// surface point (half a block out along the normal for faces). rgb = block light, a = sky light.
// Outside the fine volume (fading over its outer 12 blocks): open sky, coarse block light.
#include "common.glsl"
#include "light/light_defs.glsl"

// block light colour scale (light values are squared first for a soft falloff)
#define BLOCK_LIGHT_GAIN 2.4
// self-illumination of emissive faces
#define EMISSIVE_GAIN 5.0
// what's left in a sealed cave with no light at all
#define CAVE_AMBIENT vec3(0.018)

// The coarse volume (block light only, 4 blocks per cell) takes over beyond the fine one.
vec3 sample_coarse(FrameData f, vec3 rel, vec3 offset) {
    if ((f.coarseOrigin.w & LIGHT_ENABLED) == 0) return vec3(0.0);
    vec3 p = rel + f.camFrac.xyz + offset * float(COARSE_CELL);
    vec3 local = p - vec3(f.coarseOrigin.xyz);
    vec3 edge = min(local, vec3(COARSE_DIM * COARSE_CELL) - local);
    float fade = smoothstep(4.0, 48.0, min(edge.x, min(edge.y, edge.z)));
    if (fade <= 0.0) return vec3(0.0);
    vec3 tp = (vec3(f.camBlock.xyz & COARSE_BLOCK_MASK) + p) / float(COARSE_CELL);
    return textureLod(gTex3D[nonuniformEXT(f.coarseTex)], tp / vec3(COARSE_DIM), 0.0).rgb * fade;
}

vec4 sample_light(FrameData f, vec3 rel, vec3 offset) {
    float fade = 0.0;
    vec4 L = vec4(0.0, 0.0, 0.0, 1.0);
    if ((f.lightOrigin.w & LIGHT_ENABLED) != 0) {
        vec3 p = rel + f.camFrac.xyz + offset;            // relative to the camera block
        vec3 local = p - vec3(f.lightOrigin.xyz);
        vec3 edge = min(local, vec3(float(LIGHT_N)) - local);
        fade = smoothstep(1.0, 12.0, min(edge.x, min(edge.y, edge.z)));
        if (fade > 0.0) {
            vec3 tp = vec3(f.camBlock.xyz & LIGHT_MASK) + p;  // texel space, wraps (repeat sampler)
            L = textureLod(gTex3D[nonuniformEXT(f.lightTex)], tp / float(LIGHT_N), 0.0);
        }
    }
    if (fade >= 1.0) return L;
    // outside the fine volume: open sky (none underground), block light from the coarse volume
    float y = float(f.camBlock.y) + rel.y + f.camFrac.y + offset.y;
    float sky = smoothstep(f.skyFloor - 8.0, f.skyFloor, y);
    return mix(vec4(sample_coarse(f, rel, offset), sky), L, fade);
}

#endif
