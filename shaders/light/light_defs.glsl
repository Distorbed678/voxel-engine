#ifndef LIGHT_DEFS_GLSL
#define LIGHT_DEFS_GLSL
#define LIGHT_N 128
#define LIGHT_MASK 127
#define SKY_SHIFT 12u
#define ID_MASK 0xFFFu
#define LIGHT_ENABLED 1
// coarse (far-field) volume: COARSE_CELL blocks per cell, COARSE_DIM cells
#define COARSE_CELL 4
#define COARSE_DIM ivec3(128, 64, 128)
#define COARSE_BLOCK_MASK ivec3(511, 255, 511)   // COARSE_DIM * COARSE_CELL - 1
#endif
