#ifndef PARTICLES_COMMON_GLSL
#define PARTICLES_COMMON_GLSL
// Ambient particles and weather: simulated in a PBOX³ box that wraps around the camera (a particle
// leaving one side comes back on the other), so the camera is always surrounded by them.
#include "common.glsl"
#include "voxel.glsl"

#define PBOX 64.0
#define PBOX_MASK 63

// a particle's kind is its type's index in the type table + 1 (0 = a free slot)
#define PK_NONE 0u
#define ACTIVE_TYPES 8u

// ParticleType fields (keep in sync with worldgen::particles and GpuParticleType)
#define SHAPE_SOFT 0u
#define SHAPE_STREAK 1u
#define SHAPE_SQUARE 2u
#define SHAPE_SPARK 3u
#define MOTION_FALL 0u
#define MOTION_DRIFT 1u
#define MOTION_RISE 2u
#define MOTION_SWIRL 3u
#define MOTION_WANDER 4u
#define MOTION_FLOCK 5u
#define MOTION_ORBIT 6u
#define NEAR_ANYWHERE 0u
#define NEAR_LEAVES 1u
#define NEAR_PLANTS 2u
#define NEAR_WATER 3u
#define NEAR_GLOWING 4u
#define NEAR_GROUND 5u
#define NEAR_BLOCK 6u
#define SPAWN_OPEN_SKY 1u

struct ParticleType {
    vec4 color;    // rgb, a = opacity
    vec4 color2;   // rgb (each particle mixes the two at random), a = glow
    vec4 look;     // x = size, y = blink, z = shape, w = life
    vec4 motion;   // x = motion, y = speed, z = sway, w = wind
    uvec4 spawn;   // x = flags, y = near class, z = near block id
};
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer ParticleTypes { ParticleType t[]; };

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
    ParticleTypes types;
    uint activeKind[ACTIVE_TYPES];   // kinds spawning now (0 = unused)
    float share[ACTIVE_TYPES];   // their share of the slots
    vec2 wind;
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
