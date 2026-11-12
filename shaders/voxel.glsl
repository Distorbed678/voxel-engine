#ifndef VOXEL_GLSL
#define VOXEL_GLSL
// Shared voxel/chunk definitions. Keep in sync with crates/engine/src/world/gpu_types.rs

#define CS 32            // chunk size (cells)
#define CA 34            // chunk size with 1-cell apron on every side
#define SLOT_U16 39424u  // padded voxel slot size in u16 (34^3 = 39304)
#define NUM_STREAMS 13u  // 6 opaque dirs + 6 transparent dirs + models
#define STREAM_MODELS 12u
#define MAX_DRAW_QUADS 2048u  // quads per task workgroup (32 meshlets * 64 quads)

// Block kinds (BlockInfo.flags bits 0..3)
#define KIND_AIR 0u
#define KIND_OPAQUE 1u
#define KIND_CUTOUT 2u
#define KIND_TRANSPARENT 3u
#define KIND_MODEL 4u

// Job result flags
#define JOB_OVERFLOW 1u
#define JOB_EMPTY 2u
#define JOB_NO_MESH 4u   // region read / light-only job: skip meshing

layout(buffer_reference, scalar, buffer_reference_align = 2) buffer VoxelBuf { uint16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer ColumnBuf { uint16_t c[]; };

struct BiomeGpu {
    vec4 grass;     // rgb tint, a = fog density multiplier
    vec4 foliage;   // rgb tint, a = zone index
    vec4 water;
    vec4 mapColor;
    vec4 zoneColor;
};
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer BiomeTable { BiomeGpu b[]; };

struct BlockInfo {
    uint faces[3];   // 6 x u16 texture layers: +X,-X,+Y,-Y,+Z,-Z (first frame)
    uint flags;      // kind | tint << 4 | emission << 8 | model << 16 | fluid << 24 | rotation << 25
    uint light;      // emitted light r | g << 8 | b << 16 | attenuation << 24 | blocks-light << 28
    uint anim[3];    // 6 x u16 per face: frames - 1 (6 bits) | frames per second * 8 (10 bits)
};
#define BLOCK_FLUID (1u << 24)
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer BlockTable { BlockInfo b[]; };

struct GenJob {
    ivec3 origin;    // world block coordinate of the chunk min corner
    uint lod;
    uint editOffset;
    uint editCount;
    uint flags;      // GENJOB_*
    uint pad1;
};
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer GenJobBuf { GenJob j[]; };
#define GENJOB_LIGHT 1u   // copy the final blocks into the light volume
#define GENJOB_COARSE 2u  // summarise the blocks into the coarse light volume

struct JobResult {
    uint total;
    uint scratchBase;
    uint flags;
    uint nonAir;
    uint opaque;
    uint counts[NUM_STREAMS];
    uint pad0;
    uint pad1;
};
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer JobResultBuf { JobResult r[]; };

struct ChunkRecord {
    ivec3 origin;
    uint lodFlags;   // lod in bits 0..3, bit 8 = active
    uint quadBase;
    uint counts[NUM_STREAMS];
};
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer ChunkRecordBuf { ChunkRecord c[]; };
#define RECORD_ACTIVE 256u

struct DrawGroup {
    uint chunk;
    uint quadOffset;
    uint quadCount;
    uint dir;
};
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer DrawGroupBuf { DrawGroup g[]; };

uint vidx(ivec3 p) {
    return uint((p.y + 1) * CA * CA + (p.z + 1) * CA + (p.x + 1));
}

// A voxel value is a block id (low 12 bits) plus a state (high 4 bits): the axis of column blocks
// such as logs (0 = Y, 1 = X, 2 = Z) or the facing of stairs (0 = automatic, 1..4 = fixed).
#define BLOCK_ID(v) ((v) & 0xFFFu)
#define BLOCK_STATE(v) ((v) >> 12)

uint block_kind(BlockTable t, uint b) { return t.b[BLOCK_ID(b)].flags & 15u; }

// The face whose texture a block shows in direction `dir` (rotated column blocks show their end
// textures on their axis and a side on the Y faces).
uint face_slot(BlockTable t, uint v, uint dir) {
    uint axis = BLOCK_STATE(v) & 3u;
    if (axis != 0u && (t.b[BLOCK_ID(v)].flags & 15u) != KIND_MODEL) {
        uint la = axis == 1u ? 0u : 2u;
        if ((dir >> 1) == la) dir = 2u + (dir & 1u);
        else if ((dir >> 1) == 1u) dir = 0u;
    }
    return dir;
}

// Texture array layer of a face at animation time `time` (seconds): animated textures are
// consecutive layers.
uint face_layer(BlockTable t, uint v, uint dir, float time) {
    uint b = BLOCK_ID(v);
    dir = face_slot(t, v, dir);
    uint packed = t.b[b].faces[dir >> 1];
    uint layer = (dir & 1u) == 0u ? (packed & 0xFFFFu) : (packed >> 16);
    uint a = t.b[b].anim[dir >> 1];
    a = (dir & 1u) == 0u ? (a & 0xFFFFu) : (a >> 16);
    if (a != 0u) layer += uint(max(time, 0.0) * float(a >> 6) * 0.125) % ((a & 63u) + 1u);
    return layer;
}

// Direction helpers. dir: 0 +X, 1 -X, 2 +Y, 3 -Y, 4 +Z, 5 -Z
ivec3 dir_normal(uint dir) {
    ivec3 n = ivec3(0);
    n[dir >> 1] = (dir & 1u) == 0u ? 1 : -1;
    return n;
}

// Face-plane axes (U, V) such that cross(U, V) = +axis.
// axis X: U = Y, V = Z.   axis Y: U = Z, V = X.   axis Z: U = X, V = Y.
ivec3 axis_u(uint axis) { return axis == 0u ? ivec3(0, 1, 0) : (axis == 1u ? ivec3(0, 0, 1) : ivec3(1, 0, 0)); }
ivec3 axis_v(uint axis) { return axis == 0u ? ivec3(0, 0, 1) : (axis == 1u ? ivec3(1, 0, 0) : ivec3(0, 1, 0)); }

ivec3 slice_pos(uint axis, int s, int u, int v) {
    ivec3 p;
    p[axis] = s;
    p[(axis + 1u) % 3u] = u;   // matches axis_u
    p[(axis + 2u) % 3u] = v;   // matches axis_v
    return p;
}

// Quad packing (8 bytes):
//  w0: u(5) v(5) s(5) w-1(5) h-1(5) dir(3) lod(4, filled in by the renderer)
//  w1: block(16) ao(8) biome(8)
uvec2 pack_quad(uint u, uint v, uint s, uint w, uint h, uint dir, uint block, uint ao, uint biome) {
    uint w0 = u | (v << 5) | (s << 10) | ((w - 1u) << 15) | ((h - 1u) << 20) | (dir << 25);
    uint w1 = block | (ao << 16) | (biome << 24);
    return uvec2(w0, w1);
}

#endif
