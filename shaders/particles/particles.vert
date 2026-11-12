#version 460
// Camera-facing particle quads, 6 vertices per instance (rain is stretched along its motion).
#include "particles/particles_common.glsl"

layout(location = 0) out vec2 vUV;
layout(location = 1) flat out uint vKind;
layout(location = 2) out float vFade;
layout(location = 3) flat out float vSeed;

void main() {
    FrameData f = pc.frame.d;
    Particle q = pc.particles.p[gl_InstanceIndex];
    const vec2 corners[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    vec2 c = corners[gl_VertexIndex];
    vKind = q.kind;
    vUV = c * 0.5 + 0.5;
    vSeed = float(gl_InstanceIndex % 1024u);
    if (q.life <= 0.0 || q.kind == PK_NONE) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);   // outside the clip volume
        vFade = 0.0;
        return;
    }
    vec3 rel = box_rel(f, q.pos);
    vec3 right = vec3(f.view[0][0], f.view[1][0], f.view[2][0]);
    vec3 up = vec3(f.view[0][1], f.view[1][1], f.view[2][1]);
    float size = q.kind == PK_SNOW ? 0.06 : (q.kind == PK_ASH ? 0.05 : (q.kind == PK_EMBER ? 0.045 : (q.kind == PK_FIREFLY ? 0.07 : 0.03)));
    vec3 p;
    if (q.kind == PK_RAIN) {
        vec3 dir = normalize(q.vel + vec3(0.0, -1e-3, 0.0));
        vec3 side = normalize(cross(dir, normalize(rel)));
        p = rel + side * c.x * 0.022 + dir * c.y * 0.45;
    } else {
        p = rel + (right * c.x + up * c.y) * size;
    }
    // fade in near the camera (no huge flakes) and out towards the box edges
    float d = length(rel);
    vFade = smoothstep(0.3, 1.2, d) * (1.0 - smoothstep(PBOX * 0.38, PBOX * 0.5, d)) * clamp(q.life * 2.0, 0.0, 1.0);
    gl_Position = f.viewProj * vec4(p, 1.0);
}
