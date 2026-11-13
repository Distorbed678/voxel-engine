// Probe pass: evaluates the zone/biome layout of one layer at arbitrary X/Z positions (camera
// environment, minimap). Reuses the generator push constants: jobs -> vec2 positions, results ->
// uvec4 out (global biome and zone ids), [ownMin, ownMax] -> the heights scanned for the surface.
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer ProbeIn { vec2 p[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer ProbeOut { uvec4 r[]; };

layout(local_size_x = 64) in;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.numJobs) return;
    gSeed = pc.seed;
    gLod = 0u;
    gScale = 1.0;
    wg_init();
    ProbeIn pin = ProbeIn(uint64_t(pc.jobs));
    ProbeOut pout = ProbeOut(uint64_t(pc.results));
    vec2 xz = pin.p[i];
    WgColumn c;
    wg_column(xz, c);
    // approximate surface height: scan down from ownMax to ownMin, at most 160 steps of at least
    // 4 blocks (minimap shading)
    float h = -1e4;
    float step = max(4.0, ceil((pc.ownMax - pc.ownMin) / 160.0));
    for (float y = pc.ownMax; y > pc.ownMin; y -= step) {
        if (wg_density(vec3(xz.x, y, xz.y), c) > 0.0) { h = y; break; }
    }
    pout.r[i] = uvec4((c.biome[0] + BIOME_BASE) | ((c.zone0 + ZONE_BASE) << 16), floatBitsToUint(c.bw[0]), c.zone1 + ZONE_BASE, floatBitsToUint(h));
}
