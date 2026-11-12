#version 460
#include "common.glsl"
#include "sky.glsl"

layout(push_constant, scalar) uniform PC {
    FrameRef frame;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main() {
    FrameData f = pc.frame.d;
    vec4 ndc = vec4(vUV * 2.0 - 1.0, 1e-4, 1.0);
    vec4 wp = f.invViewProj * ndc;
    vec3 dir = normalize(wp.xyz / wp.w);
    outColor = vec4(sky_with_sun(f, dir), 1.0);
}
