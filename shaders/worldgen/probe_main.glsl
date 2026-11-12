// Probe pass: evaluates the zone/biome layout at arbitrary X/Z positions (camera environment,
// minimap). Reuses the generator push constants: jobs -> vec2 positions, results -> uvec4 out.
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
    // approximate surface height: scan down from 400 in 4-block steps (minimap shading)
    float h = -1e4;
    if (i < 8192u) {
        for (int y = 400; y > -128; y -= 4) {
            if (wg_density(vec3(xz.x, float(y), xz.y), c) > 0.0) { h = float(y); break; }
        }
    }
    pout.r[i] = uvec4(c.biome[0] | (c.zone0 << 16), floatBitsToUint(c.bw[0]), c.zone1, floatBitsToUint(h));
}
