#ifndef PARTICLES_COMMON_GLSL
#define PARTICLES_COMMON_GLSL
// Ambient particles and weather: simulated in a PBOX³ box that wraps around the camera (a particle
// leaving one side comes back on the other), so the camera is always surrounded by them.
#include "common.glsl"
#include "voxel.glsl"

#define PBOX 64.0
#define PBOX_MASK 63

#define PK_NONE 0u
#define PK_SNOW 1u
#define PK_RAIN 2u
#define PK_ASH 3u
#define PK_EMBER 4u
#define PK_FIREFLY 5u
#define PK_POLLEN 6u

struct Particle {
    vec3 pos;      // box space [0, PBOX)
    float life;    // seconds left; <= 0 = free slot
    vec3 vel;
    uint kind;
};
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer ParticleBuf { Particle p[]; };

layout(push_constant, scalar) uniform PC {
    FrameRef frame;
    ParticleBuf particles;
    BlockTable blocks;
    vec4 weights0;      // share of the slots: snow, rain, ash, ember
    vec4 weights1;      // firefly, pollen, wind x, wind z
    uint count;
    uint lightBlocks;   // bindless storage index of the light volume's block image
    uint frameIndex;
    float dt;
} pc;

// position relative to the camera, wrapped into [-PBOX/2, PBOX/2)
vec3 box_rel(FrameData f, vec3 pos) {
    vec3 camMod = vec3(f.camBlock.xyz & PBOX_MASK) + f.camFrac.xyz;
    vec3 d = pos - camMod;
    return d - PBOX * floor(d / PBOX + 0.5);
}

float phash(uint a, uint b) {
    return float(hash_u32(a * 747796405u + b * 2891336453u + 0x9e3779b9u) & 0xFFFFFFu) / 16777216.0;
}

#endif
