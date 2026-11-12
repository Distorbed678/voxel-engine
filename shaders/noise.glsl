#ifndef NOISE_GLSL
#define NOISE_GLSL
// Seeded, hash-based gradient noise (no permutation tables). All functions are pure and
// deterministic so the GPU generator always reproduces the same world for a seed.

uvec3 pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

uint hash3i(ivec3 c, uint seed) {
    return pcg3d(uvec3(c) ^ uvec3(seed, seed * 0x9E3779B9u, seed * 0x85EBCA6Bu)).x;
}

uint hash2i(ivec2 c, uint seed) {
    return pcg3d(uvec3(uvec2(c), seed)).x;
}

float hash_to_unit(uint h) { return float(h >> 8) * (1.0 / 16777216.0); }

float rand2(ivec2 c, uint seed) { return hash_to_unit(hash2i(c, seed)); }
float rand3(ivec3 c, uint seed) { return hash_to_unit(hash3i(c, seed)); }

vec3 rand3v(ivec3 c, uint seed) {
    uvec3 h = pcg3d(uvec3(c) ^ uvec3(seed * 747796405u, seed, seed * 2891336453u));
    return vec3(h >> 8) * (1.0 / 16777216.0);
}

vec2 rand2v(ivec2 c, uint seed) {
    uvec3 h = pcg3d(uvec3(uvec2(c), seed ^ 0xA511E9B3u));
    return vec2(h.xy >> 8) * (1.0 / 16777216.0);
}

vec3 grad3(ivec3 c, uint seed) {
    // 12 cube-edge gradients (classic, unbiased)
    uint h = hash3i(c, seed) % 12u;
    float u = h < 8u ? 1.0 : 0.0;
    vec3 g;
    if (h < 4u) g = vec3((h & 1u) == 0u ? 1.0 : -1.0, (h & 2u) == 0u ? 1.0 : -1.0, 0.0);
    else if (h < 8u) g = vec3((h & 1u) == 0u ? 1.0 : -1.0, 0.0, (h & 2u) == 0u ? 1.0 : -1.0);
    else g = vec3(0.0, (h & 1u) == 0u ? 1.0 : -1.0, (h & 2u) == 0u ? 1.0 : -1.0);
    return g;
}

vec2 grad2(ivec2 c, uint seed) {
    uint h = hash2i(c, seed);
    float a = float(h & 0xFFFFu) * (6.28318530718 / 65536.0);
    return vec2(cos(a), sin(a));
}

// 3D simplex noise in [-1, 1]
float simplex3(vec3 p, uint seed) {
    const float F3 = 1.0 / 3.0;
    const float G3 = 1.0 / 6.0;
    vec3 s = floor(p + dot(p, vec3(F3)));
    vec3 x0 = p - s + dot(s, vec3(G3));
    vec3 e = step(vec3(0.0), x0 - x0.yzx);
    vec3 i1 = e * (1.0 - e.zxy);
    vec3 i2 = 1.0 - e.zxy * (1.0 - e);
    vec3 x1 = x0 - i1 + G3;
    vec3 x2 = x0 - i2 + 2.0 * G3;
    vec3 x3 = x0 - 1.0 + 3.0 * G3;
    ivec3 is = ivec3(s);
    vec4 w = vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3));
    w = max(0.6 - w, 0.0);
    vec4 d = vec4(dot(grad3(is, seed), x0),
                  dot(grad3(is + ivec3(i1), seed), x1),
                  dot(grad3(is + ivec3(i2), seed), x2),
                  dot(grad3(is + ivec3(1), seed), x3));
    w *= w;
    w *= w;
    return clamp(32.0 * dot(w, d), -1.0, 1.0);
}

// 2D simplex noise in [-1, 1]
float simplex2(vec2 p, uint seed) {
    const float K1 = 0.366025404;
    const float K2 = 0.211324865;
    vec2 i = floor(p + (p.x + p.y) * K1);
    vec2 a = p - i + (i.x + i.y) * K2;
    float m = step(a.y, a.x);
    vec2 o = vec2(m, 1.0 - m);
    vec2 b = a - o + K2;
    vec2 c = a - 1.0 + 2.0 * K2;
    vec3 h = max(0.5 - vec3(dot(a, a), dot(b, b), dot(c, c)), 0.0);
    ivec2 ii = ivec2(i);
    vec3 n = h * h * h * h * vec3(dot(a, grad2(ii, seed)),
                                  dot(b, grad2(ii + ivec2(o), seed)),
                                  dot(c, grad2(ii + ivec2(1), seed)));
    return clamp(dot(n, vec3(70.0)), -1.0, 1.0);
}

// Value noise (cheap, blobby) in [-1, 1]
float value3(vec3 p, uint seed) {
    ivec3 i = ivec3(floor(p));
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float n000 = rand3(i, seed), n100 = rand3(i + ivec3(1, 0, 0), seed);
    float n010 = rand3(i + ivec3(0, 1, 0), seed), n110 = rand3(i + ivec3(1, 1, 0), seed);
    float n001 = rand3(i + ivec3(0, 0, 1), seed), n101 = rand3(i + ivec3(1, 0, 1), seed);
    float n011 = rand3(i + ivec3(0, 1, 1), seed), n111 = rand3(i + ivec3(1, 1, 1), seed);
    float r = mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
                  mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
    return r * 2.0 - 1.0;
}

// ---- fractals ----------------------------------------------------------------------------
float fbm2(vec2 p, uint seed, int octaves, float lacunarity, float gain) {
    float sum = 0.0, amp = 1.0, norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * simplex2(p, seed + uint(i) * 1013u);
        norm += amp;
        p *= lacunarity;
        amp *= gain;
    }
    return sum / max(norm, 1e-6);
}

float fbm3(vec3 p, uint seed, int octaves, float lacunarity, float gain) {
    float sum = 0.0, amp = 1.0, norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * simplex3(p, seed + uint(i) * 1013u);
        norm += amp;
        p *= lacunarity;
        amp *= gain;
    }
    return sum / max(norm, 1e-6);
}

float ridged2(vec2 p, uint seed, int octaves, float lacunarity, float gain) {
    float sum = 0.0, amp = 1.0, norm = 0.0, prev = 1.0;
    for (int i = 0; i < octaves; ++i) {
        float n = 1.0 - abs(simplex2(p, seed + uint(i) * 1013u));
        n *= n;
        sum += amp * n * prev;
        prev = n;
        norm += amp;
        p *= lacunarity;
        amp *= gain;
    }
    return (sum / max(norm, 1e-6)) * 2.0 - 1.0;
}

float ridged3(vec3 p, uint seed, int octaves, float lacunarity, float gain) {
    float sum = 0.0, amp = 1.0, norm = 0.0, prev = 1.0;
    for (int i = 0; i < octaves; ++i) {
        float n = 1.0 - abs(simplex3(p, seed + uint(i) * 1013u));
        n *= n;
        sum += amp * n * prev;
        prev = n;
        norm += amp;
        p *= lacunarity;
        amp *= gain;
    }
    return (sum / max(norm, 1e-6)) * 2.0 - 1.0;
}

float billow2(vec2 p, uint seed, int octaves, float lacunarity, float gain) {
    float sum = 0.0, amp = 1.0, norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * (abs(simplex2(p, seed + uint(i) * 1013u)) * 2.0 - 1.0);
        norm += amp;
        p *= lacunarity;
        amp *= gain;
    }
    return sum / max(norm, 1e-6);
}

float billow3(vec3 p, uint seed, int octaves, float lacunarity, float gain) {
    float sum = 0.0, amp = 1.0, norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * (abs(simplex3(p, seed + uint(i) * 1013u)) * 2.0 - 1.0);
        norm += amp;
        p *= lacunarity;
        amp *= gain;
    }
    return sum / max(norm, 1e-6);
}

// ---- cellular ----------------------------------------------------------------------------
// Returns (F1, F2, cell hash as float bits) for jittered 2D Worley noise.
vec3 worley2(vec2 p, uint seed, float jitter) {
    ivec2 c = ivec2(floor(p));
    float f1 = 1e9, f2 = 1e9;
    uint id = 0u;
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x) {
        ivec2 cc = c + ivec2(x, y);
        vec2 fp = vec2(cc) + 0.5 + (rand2v(cc, seed) - 0.5) * jitter;
        float d = distance(p, fp);
        if (d < f1) { f2 = f1; f1 = d; id = hash2i(cc, seed ^ 0x51ED270Bu); }
        else if (d < f2) { f2 = d; }
    }
    return vec3(f1, f2, uintBitsToFloat(id & 0x007FFFFFu | 0x3F800000u) - 1.0);
}

vec3 worley3(vec3 p, uint seed, float jitter) {
    ivec3 c = ivec3(floor(p));
    float f1 = 1e9, f2 = 1e9;
    uint id = 0u;
    for (int z = -1; z <= 1; ++z)
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x) {
        ivec3 cc = c + ivec3(x, y, z);
        vec3 fp = vec3(cc) + 0.5 + (rand3v(cc, seed) - 0.5) * jitter;
        float d = distance(p, fp);
        if (d < f1) { f2 = f1; f1 = d; id = hash3i(cc, seed ^ 0x51ED270Bu); }
        else if (d < f2) { f2 = d; }
    }
    return vec3(f1, f2, uintBitsToFloat(id & 0x007FFFFFu | 0x3F800000u) - 1.0);
}

float smoothmin(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / max(k, 1e-6);
    return min(a, b) - h * h * k * 0.25;
}

float smoothmax(float a, float b, float k) {
    return -smoothmin(-a, -b, k);
}

#endif
