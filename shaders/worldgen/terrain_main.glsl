// Terrain pass main. Expects the generated code to define (see crates/worldgen/src/compile.rs):
//   struct WgColumn; void wg_init(); void wg_column(vec2 xz, out WgColumn c);
//   float wg_density(vec3 p, in WgColumn c);                                   (> 0 = solid)
//   uint  wg_material(vec3 p, float depth, float airAbove, bool skyOpen, in WgColumn c);
//   uint  wg_fluid(vec3 p, bool skyOpen, in WgColumn c);
//   uint  wg_decorate(vec3 p, uint ground, in WgColumn c);
//   uint  wg_main_biome(in WgColumn c);
//
// One invocation = one (x, z) column of 34 cells (chunk + apron), generated top-down so we know
// how deep below the surface each solid cell is (material layering) and whether the cell can
// see the sky (oceans vs. dry caves).

#ifndef WG_LOOKAHEAD
#define WG_LOOKAHEAD 8
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    uint jobIdx = gl_WorkGroupID.z;
    GenJob job = pc.jobs.j[jobIdx];
    gSeed = pc.seed;
    gLod = job.lod;
    gScale = float(1 << job.lod);
    wg_init();
    ivec2 lc = ivec2(gl_GlobalInvocationID.xy);
    bool valid = lc.x < CA && lc.y < CA;

    uint nonAir = 0u;
    uint opaque = 0u;

    if (valid) {
        int scale = 1 << job.lod;
        int half_ = scale >> 1;
        vec2 xz = vec2(job.origin.xz + (lc - 1) * scale + half_) + 0.5;
        WgColumn col;
        wg_column(xz, col);
        pc.columns.c[jobIdx * uint(CA * CA) + uint(lc.y * CA + lc.x)] = uint16_t(wg_main_biome(col));

        uint base = jobIdx * SLOT_U16;
        float depth = 0.0;
        float airAbove = 0.0;
        bool skyOpen = true;
        uint prevBlock = 0u;
        int prevLy = -1;
        // single loop (one wg_density call site keeps the generated shader small):
        // ly >= CA is look-ahead above the chunk (only tracks depth / sky visibility)
        for (int ly = CA - 1 + WG_LOOKAHEAD; ly >= 0; --ly) {
            vec3 p = vec3(xz.x, float(job.origin.y + (ly - 1) * scale + half_) + 0.5, xz.y);
            float d = wg_density(p, col);
            bool solid = d > 0.0;
            if (ly < CA) {
                uint b = solid ? wg_material(p, depth, airAbove, skyOpen, col) : wg_fluid(p, skyOpen, col);
                if (solid && depth == 0.0 && prevLy >= 0 && prevBlock == 0u && job.lod == 0u) {
                    uint deco = wg_decorate(p + vec3(0.0, 1.0, 0.0), b, col);
                    if (deco != 0u) {
                        pc.voxels.v[base + uint(prevLy * CA * CA + lc.y * CA + lc.x)] = uint16_t(deco);
                        nonAir += 1u;
                    }
                }
                pc.voxels.v[base + uint(ly * CA * CA + lc.y * CA + lc.x)] = uint16_t(b);
                if (b != 0u) nonAir += 1u;
                if (solid && b != 0u) opaque += 1u;
                prevBlock = b;
                prevLy = ly;
            }
            if (solid) { depth += float(scale); airAbove = 0.0; skyOpen = false; }
            else { depth = 0.0; airAbove += float(scale); }
        }
    }

    uint sNonAir = subgroupAdd(nonAir);
    uint sOpaque = subgroupAdd(opaque);
    if (subgroupElect()) {
        atomicAdd(pc.results.r[jobIdx].nonAir, sNonAir);
        atomicAdd(pc.results.r[jobIdx].opaque, sOpaque);
    }
}
