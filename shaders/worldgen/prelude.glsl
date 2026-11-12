// Included before the generated world-generation code.
#include "common.glsl"
#include "noise.glsl"
#include "voxel.glsl"
#include "generated/blocks.glsl"

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer EditBuf { uvec2 e[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer ParamBuf { float p[]; };

layout(push_constant, scalar) uniform GenPC {
    GenJobBuf jobs;
    VoxelBuf voxels;
    JobResultBuf results;
    EditBuf edits;
    ParamBuf params;     // live-tweakable node parameters (no recompile needed)
    uint numJobs;
    uint seed;
    uint editCount;
    uint pad;
} pc;

// Globals set by the main template before calling into generated code.
uint gSeed;
uint gLod;      // LOD level of the chunk being generated (0 = full detail)
float gScale;   // cell size in blocks (1 << gLod)

// Node parameters are read through this so dragging a slider never recompiles the shader.
float P(uint i) { return pc.params.p[i]; }

// Helpers available to generated code -------------------------------------------------
float remap01(float x) { return x * 0.5 + 0.5; }

float curve_eval(uint base, uint count, float x) {
    // Piecewise-linear curve stored in params: count points of (x, y), sorted by x.
    if (count == 0u) return x;
    float x0 = P(base), y0 = P(base + 1u);
    if (x <= x0) return y0;
    for (uint i = 1u; i < count; ++i) {
        float x1 = P(base + i * 2u), y1 = P(base + i * 2u + 1u);
        if (x <= x1) {
            float t = (x - x0) / max(x1 - x0, 1e-6);
            // smooth (monotone-ish) interpolation for nicer terrain
            t = t * t * (3.0 - 2.0 * t);
            return mix(y0, y1, t);
        }
        x0 = x1; y0 = y1;
    }
    return y0;
}
