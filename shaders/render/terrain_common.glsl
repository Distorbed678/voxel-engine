#ifndef TERRAIN_COMMON_GLSL
#define TERRAIN_COMMON_GLSL
#extension GL_EXT_mesh_shader : require
#include "common.glsl"
#include "voxel.glsl"

layout(push_constant, scalar) uniform TerrainPC {
    FrameRef frame;
    ChunkRecordBuf records;
    DrawGroupBuf draws;
    UVec2Buf quads;
    BlockTable blocks;
    U64Buf pick;
    uint flags;
    uint pad0;
} pc;

struct TaskPayload {
    uint chunk;
    uint quadOffset;
    uint quadCount;
    uint group;
};

#endif
