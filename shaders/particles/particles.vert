#version 460
// Camera-facing particle quads, 6 vertices per instance, shaped by the particle's type: streaks
// stretch along their motion, squares tumble. Lit by the light volume where they are.
#include "particles/particles_common.glsl"
#include "light/light_sample.glsl"

layout(location = 0) out vec2 vUV;
layout(location = 1) flat out uint vKind;
layout(location = 2) out float vFade;
layout(location = 3) flat out float vSeed;
layout(location = 4) out vec4 vLight;   // block light (rgb, gain applied) and sky visibility

void main() {
    FrameData f = pc.frame.d;
    Particle q = pc.particles.p[gl_InstanceIndex];
    const vec2 corners[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    vec2 c = corners[gl_VertexIndex];
    vKind = q.kind;
    vUV = c * 0.5 + 0.5;
    vSeed = float(gl_InstanceIndex % 1024u);
    vLight = vec4(0.0);
    if (q.life <= 0.0 || q.kind == PK_NONE) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // outside the clip volume
        vFade = 0.0;
        return;
    }
    ParticleType T = pc.types.t[q.kind - 1u];
    float size = T.look.x;
    uint shape = uint(T.look.z);
    vec3 rel = box_rel(f, q.pos);
    vec3 right = vec3(f.view[0][0], f.view[1][0], f.view[2][0]);
    vec3 up = vec3(f.view[0][1], f.view[1][1], f.view[2][1]);
    vec3 p;
    if (shape == SHAPE_STREAK) {
        vec3 dir = normalize(q.vel + vec3(0.0, -1e-3, 0.0));
        vec3 side = normalize(cross(dir, normalize(rel)));
        p = rel + side * c.x * size * 0.75 + dir * c.y * size * 15.0;
    } else if (shape == SHAPE_SQUARE) {
        // tumbling: spin in the view plane and flip (squash one side) over time
        float a = vSeed * 0.37 + f.time * (1.2 + fract(vSeed * 0.131) * 2.0);
        vec2 k = vec2(c.x * abs(cos(f.time * 1.7 + vSeed)) + 0.15 * c.x, c.y);
        vec2 r = vec2(k.x * cos(a) - k.y * sin(a), k.x * sin(a) + k.y * cos(a));
        p = rel + (right * r.x + up * r.y) * size;
    } else {
        p = rel + (right * c.x + up * c.y) * size;
    }
    vec4 L = sample_light(f, rel, vec3(0.0));
    vLight = vec4(L.rgb * L.rgb * BLOCK_LIGHT_GAIN + CAVE_AMBIENT, L.a);
    // fade in near the camera (no huge flakes) and out towards the box edges
    float d = length(rel);
    vFade = smoothstep(0.3, 1.2, d) * (1.0 - smoothstep(PBOX * 0.38, PBOX * 0.5, d)) * clamp(q.life * 2.0, 0.0, 1.0);
    gl_Position = f.viewProj * vec4(p, 1.0);
}
