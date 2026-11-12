#ifndef LIGHT_COMMON_GLSL
#define LIGHT_COMMON_GLSL
// Light volume: a toroidal LIGHT_N^3 block-resolution volume around the camera.
// Keep LIGHT_N in sync with crates/engine/src/world/light.rs.
#include "common.glsl"
#include "voxel.glsl"
#include "light/light_defs.glsl"

// block volume: block id | sky source level << SKY_SHIFT (written by fill.comp + sky.comp)
layout(set = 0, binding = 1, r16ui) uniform uimage3D gLightBlocks[];
// light volume: rgb block light, a sky light (ping-pong pair, written by propagate.comp)
layout(set = 0, binding = 1, rgba8) uniform image3D gLightImg[];

vec3 light_emit(BlockTable t, uint id) {
    uint l = t.b[id].light;
    return vec3(float(l & 255u), float((l >> 8) & 255u), float((l >> 16) & 255u)) / 255.0;
}
uint light_att(BlockTable t, uint id) { return (t.b[id].light >> 24) & 15u; }
bool light_blocked(BlockTable t, uint id) { return ((t.b[id].light >> 28) & 1u) != 0u; }

#endif
