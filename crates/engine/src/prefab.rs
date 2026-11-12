//! Prefabs: saved block selections. Usable from the clipboard and from the world generator's
//! "Prefab" prop node (uploaded to the GPU as one table + one block buffer).

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use glam::IVec3;
use serde::{Deserialize, Serialize};
use vkcore::{Buffer, VkContext};

use crate::blocks::BlockRegistry;

#[derive(Serialize, Deserialize)]
struct PrefabFile {
    size: [i32; 3],
    palette: Vec<String>,
    /// run-length encoded palette indices: (run, index)
    runs: Vec<(u32, u16)>,
}

#[derive(Clone)]
pub struct Prefab {
    pub name: String,
    pub size: IVec3,
    /// block ids, x fastest, then z, then y
    pub blocks: Vec<u16>,
    /// The cell that lands where it is pasted (a prop's anchor; zero for copies).
    pub origin: IVec3,
}

impl Prefab {
    pub fn get(&self, p: IVec3) -> u16 {
        if p.cmplt(IVec3::ZERO).any() || p.cmpge(self.size).any() {
            return 0;
        }
        self.blocks[((p.y * self.size.z + p.z) * self.size.x + p.x) as usize]
    }

    /// Rotate 90 degrees around Y (+X towards +Z); block states turn with it.
    pub fn rotated(&self, reg: &BlockRegistry) -> Prefab {
        let s = self.size;
        let ns = IVec3::new(s.z, s.y, s.x);
        let mut out = vec![0u16; self.blocks.len()];
        for y in 0..s.y {
            for z in 0..s.z {
                for x in 0..s.x {
                    let (nx, nz) = (s.z - 1 - z, x);
                    out[((y * ns.z + nz) * ns.x + nx) as usize] = reg.rotated_y(self.get(IVec3::new(x, y, z)), 1);
                }
            }
        }
        let o = self.origin;
        Prefab { name: self.name.clone(), size: ns, blocks: out, origin: IVec3::new(s.z - 1 - o.z, o.y, o.x) }
    }
}

pub struct PrefabLibrary {
    pub prefabs: Vec<Prefab>,
    dir: PathBuf,
    pub table: Option<Buffer>,
    pub data: Option<Buffer>,
    pub gpu_dirty: bool,
    pub revision: u64,
}

impl PrefabLibrary {
    pub fn load(dir: PathBuf, reg: &BlockRegistry) -> Self {
        let mut prefabs = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.extension().map(|e| e == "ron").unwrap_or(false)).collect();
            files.sort();
            for f in files {
                let name = f.file_stem().unwrap().to_string_lossy().to_string();
                match std::fs::read_to_string(&f).map_err(anyhow::Error::from).and_then(|s| ron::from_str::<PrefabFile>(&s).map_err(Into::into)) {
                    Ok(pf) => {
                        let ids: Vec<u16> = pf.palette.iter().map(|n| reg.parse_value(n).unwrap_or(0)).collect();
                        let mut blocks = Vec::new();
                        for (run, idx) in pf.runs {
                            blocks.extend(std::iter::repeat_n(ids.get(idx as usize).copied().unwrap_or(0), run as usize));
                        }
                        let size = IVec3::from(pf.size);
                        blocks.resize((size.x * size.y * size.z).max(0) as usize, 0);
                        prefabs.push(Prefab { name, size, blocks, origin: IVec3::ZERO });
                    }
                    Err(e) => log::warn!("prefab {}: {e}", f.display()),
                }
            }
        }
        if prefabs.is_empty() {
            prefabs.push(default_ruin(reg));
        }
        log::info!("loaded {} prefab(s)", prefabs.len());
        Self { prefabs, dir, table: None, data: None, gpu_dirty: true, revision: 1 }
    }

    pub fn save(&mut self, p: Prefab, reg: &BlockRegistry) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let mut palette: Vec<String> = Vec::new();
        let mut index: HashMap<u16, u16> = HashMap::new();
        let mut runs: Vec<(u32, u16)> = Vec::new();
        for &b in &p.blocks {
            let i = *index.entry(b).or_insert_with(|| {
                palette.push(reg.value_name(b));
                (palette.len() - 1) as u16
            });
            match runs.last_mut() {
                Some((n, j)) if *j == i => *n += 1,
                _ => runs.push((1, i)),
            }
        }
        let file = PrefabFile { size: p.size.to_array(), palette, runs };
        std::fs::write(self.dir.join(format!("{}.ron", p.name)), ron::ser::to_string(&file)?)?;
        if let Some(existing) = self.prefabs.iter_mut().find(|e| e.name == p.name) {
            *existing = p;
        } else {
            self.prefabs.push(p);
        }
        self.gpu_dirty = true;
        self.revision += 1;
        Ok(())
    }

    pub fn names(&self) -> Vec<String> {
        self.prefabs.iter().map(|p| p.name.clone()).collect()
    }

    /// name -> (index, size) for the world-generator compiler
    pub fn compile_info(&self) -> HashMap<String, (u32, [i32; 3])> {
        self.prefabs.iter().enumerate().map(|(i, p)| (p.name.clone(), (i as u32, p.size.to_array()))).collect()
    }

    /// (Re)upload to the GPU. Safe to call only when the GPU is not using the old buffers
    /// (the caller waits for idle; prefab edits are rare).
    pub fn upload(&mut self, ctx: &VkContext) -> Result<()> {
        if !self.gpu_dirty {
            return Ok(());
        }
        ctx.wait_idle();
        let mut table: Vec<[u32; 4]> = Vec::new();
        let mut data: Vec<u16> = Vec::new();
        for p in &self.prefabs {
            table.push([data.len() as u32, p.size.x as u32, p.size.y as u32, p.size.z as u32]);
            data.extend_from_slice(&p.blocks);
        }
        if table.is_empty() {
            table.push([0; 4]);
        }
        data.push(0);
        if data.len() % 2 == 1 {
            data.push(0);
        }
        for b in [self.table.take(), self.data.take()].into_iter().flatten() {
            let mut b = b;
            b.destroy(ctx);
        }
        let mut t = Buffer::upload(ctx, "prefab_table", (table.len() * 16) as u64)?;
        t.write(0, &table);
        let mut d = Buffer::upload(ctx, "prefab_data", (data.len() * 2) as u64)?;
        d.write(0, &data);
        self.table = Some(t);
        self.data = Some(d);
        self.gpu_dirty = false;
        Ok(())
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        for b in [self.table.take(), self.data.take()].into_iter().flatten() {
            let mut b = b;
            b.destroy(ctx);
        }
    }
}

/// A small built-in ruin so the Prefab node has something to place out of the box.
fn default_ruin(reg: &BlockRegistry) -> Prefab {
    let size = IVec3::new(7, 5, 7);
    let mut blocks = vec![0u16; (size.x * size.y * size.z) as usize];
    let cobble = reg.id("cobblestone");
    let mossy = reg.id("mossy_cobblestone");
    let bricks = reg.id("stone_bricks");
    let slab = reg.id("stone_slab");
    let lantern = reg.id("lantern");
    let mut set = |x: i32, y: i32, z: i32, b: u16| blocks[((y * size.z + z) * size.x + x) as usize] = b;
    for z in 0..7 {
        for x in 0..7 {
            set(x, 0, z, if (x + z) % 3 == 0 { mossy } else { cobble });
            let wall = x == 0 || z == 0 || x == 6 || z == 6;
            if wall {
                let h = 1 + ((x * 7 + z * 13) % 4);
                for y in 1..h.min(5) {
                    set(x, y, z, if y == 1 { mossy } else { bricks });
                }
            }
        }
    }
    set(3, 1, 3, slab);
    set(3, 2, 3, lantern);
    // doorway
    for x in 2..5 {
        for y in 1..5 {
            set(x, y, 0, 0);
        }
    }
    Prefab { name: "ruin".into(), size, blocks, origin: IVec3::ZERO }
}
