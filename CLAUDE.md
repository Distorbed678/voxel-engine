# Voxel Engine: notes for Claude

## Working with the owner

- **Questions and feature suggestions go through the question tool (`AskUserQuestion`)**, never as
  open questions in chat. This covers anything that needs the owner's decision, and any idea
  for a feature to implement. Batch related questions into one call.
- `docs/PLAN.md` is the source of truth for decisions, status and the roadmap. Follow the work
  order there, respect choices recorded as "not selected", and update it when a milestone lands
  or a decision changes.
- Commit (and push to `origin main`) whenever a milestone is done and verified; no need to ask.
  The public repo is https://github.com/Distorbed678/voxel-engine.

## Build and verify

- `cargo build --release` and `cargo test --workspace` must stay green.
- Check visuals with offscreen screenshots and validation on, and require 0 validation errors:
  `./target/release/voxel-engine --headless --validate --frames 120 --screenshot out.png --pos x,y,z --look yaw,pitch --time 0..1`
- Interactive input (mouse/keyboard, node editor) can't be exercised headlessly by hand; prefer
  automated input scripts (see PLAN.md) and list what still needs a hand test.
