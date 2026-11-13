#ifndef SKY_GLSL
#define SKY_GLSL
#include "common.glsl"

// Analytic stylised sky. Colours for the current time of day are computed on the CPU.
vec3 sky_color(FrameData f, vec3 dir) {
    vec3 sunDir = f.sunDir.xyz;
    float y = dir.y;
    float horizonBlend = pow(1.0 - clamp(y, 0.0, 1.0), 4.0);
    vec3 col = mix(f.skyZenith.rgb, f.skyHorizon.rgb, horizonBlend);
    // below the horizon: haze with a hint of the ground (what lies beyond the streamed world, seen
    // through kilometres of air)
    if (y < 0.0) {
        col = mix(f.skyHorizon.rgb, mix(f.skyHorizon.rgb, f.groundColor.rgb, 0.3) * 0.85, clamp(-y * 3.0, 0.0, 1.0));
    }
    float mu = dot(dir, sunDir);
    float sunUp = smoothstep(-0.15, 0.1, sunDir.y);
    // Mie-ish glow around the sun, warmer near the horizon
    vec3 glowCol = mix(vec3(1.0, 0.45, 0.15), f.sunColor.rgb, clamp(sunDir.y * 3.0, 0.0, 1.0));
    col += glowCol * pow(max(mu, 0.0), 12.0) * 0.6 * sunUp;
    col += glowCol * pow(max(mu, 0.0), 2.0) * 0.12 * horizonBlend * sunUp;
    // underground (the camera below the sky floor) there is no sky: fog and haze fade to dark
    float camY = float(f.camBlock.y) + f.camFrac.y;
    col = mix(col, vec3(0.012, 0.014, 0.018), smoothstep(f.skyFloor, f.skyFloor - 48.0, camY));
    return col;
}

vec3 sky_with_sun(FrameData f, vec3 dir) {
    vec3 col = sky_color(f, dir);
    float mu = dot(dir, f.sunDir.xyz);
    float disk = smoothstep(0.9995, 0.99975, mu);
    col += f.sunColor.rgb * disk * 40.0 * smoothstep(-0.05, 0.02, f.sunDir.y);
    // moon (opposite the sun)
    float moon = smoothstep(0.9993, 0.9996, dot(dir, -f.sunDir.xyz));
    col += vec3(0.8, 0.85, 1.0) * moon * 2.0 * smoothstep(0.0, -0.1, f.sunDir.y);
    // stars
    float night = smoothstep(0.05, -0.2, f.sunDir.y);
    if (night > 0.0 && dir.y > 0.0) {
        vec3 d = dir * 300.0;
        ivec3 cell = ivec3(floor(d));
        uint h = hash_u32(uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u ^ uint(cell.z) * 83492791u);
        if ((h & 1023u) < 3u) {
            vec3 sp = fract(d) - 0.5;
            float star = smoothstep(0.25, 0.0, length(sp));
            col += vec3(star) * night * float((h >> 10) & 255u) / 255.0 * 3.0;
        }
    }
    return col;
}

// Height + distance fog that fades geometry into the sky colour.
vec3 apply_fog(FrameData f, vec3 color, vec3 rel, vec3 viewDir) {
    if ((f.flags & FLAG_FOG) == 0u) return color;
    float dist = length(rel);
    float density = f.fogColor.a;
    float camY = float(f.camBlock.y) + f.camFrac.y;
    float falloff = max(f.fogParams.x, 1e-5);
    // analytic height fog integral along the ray
    float h0 = camY - f.fogParams.z;
    float dy = rel.y;
    float heightTerm = exp(-falloff * max(h0, -50.0));
    float integral = abs(dy) > 0.01
        ? heightTerm * (1.0 - exp(-falloff * dy)) / (falloff * dy)
        : heightTerm;
    float amount = clamp(1.0 - exp(-density * max(dist - f.fogParams.y, 0.0) * clamp(integral, 0.0, 4.0)), 0.0, 1.0);
    vec3 sky = sky_color(f, viewDir);
    vec3 fogCol = mix(f.fogColor.rgb, sky, 0.65);
    color = mix(color, fogCol, amount);
    // distance haze: the far LOD rings melt into exactly the sky behind them, so the edge of the
    // streamed world never shows (seen from high above, it is a square)
    float haze = smoothstep(f.fogParams.w * 0.55, f.fogParams.w, dist);
    return mix(color, sky, haze);
}

#endif
