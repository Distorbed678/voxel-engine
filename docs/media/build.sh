#!/usr/bin/env bash
# Regenerate the README media: stills (JPEG), an animated WebP preview and an MP4 flythrough.
# Needs a Vulkan GPU, ImageMagick and ffmpeg (libwebp, libx264). Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/../.."
cargo build --release
out=target/media
rm -rf "$out" && mkdir -p "$out"
./target/release/voxel-engine --size 1920x1080 --script docs/media/stills.vxs
for f in hero badlands taiga glacier lava night editor cave_lush cave_crystal rivers; do
    magick "$out/$f.png" -quality 86 "docs/media/$f.jpg"
done
magick montage "$out"/view_zones.png "$out"/view_biomes.png "$out"/view_lod.png "$out"/view_meshlets.png \
    -tile 2x2 -geometry 960x540+4+4 -background '#0d1117' -quality 86 docs/media/debug_views.jpg
./target/release/voxel-engine --size 1280x720 --script docs/media/flythrough.vxs
ffmpeg -y -loglevel error -framerate 60 -i "$out/fly/frame_%05d.png" -vf fps=30 -c:v libx264 -preset slow -crf 27 \
    -pix_fmt yuv420p -movflags +faststart docs/media/flythrough.mp4
ffmpeg -y -loglevel error -framerate 60 -i "$out/fly/frame_%05d.png" -vf "fps=15,scale=720:-1:flags=lanczos" \
    -c:v libwebp_anim -quality 55 -compression_level 6 -loop 0 docs/media/flythrough.webp
ls -la docs/media
