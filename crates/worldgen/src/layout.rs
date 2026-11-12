//! Automatic graph layout for the node editor: columns by distance to the output (data flows left
//! to right), nodes ordered within a column by the average height of the nodes they feed
//! (fewer crossings), stacked with estimated node heights so nothing overlaps.

use std::collections::HashMap;

use crate::graph::Graph;
use crate::nodes::{self, ParamKind};

/// Horizontal distance between columns (node width + gap).
pub const COLUMN_WIDTH: f32 = 330.0;
/// Vertical gap between nodes in a column.
pub const ROW_GAP: f32 = 36.0;

/// Approximate on-screen size of a node (the editor draws one row per pin and parameter).
pub fn node_size(kind: &str) -> [f32; 2] {
    let Some(def) = nodes::def(kind) else { return [200.0, 80.0] };
    if kind == "comment" {
        return [0.0, 0.0];
    }
    let mut h = 34.0 + 20.0 * def.inputs.len() as f32 + 20.0 * def.outputs.len() as f32;
    for p in def.params {
        h += match p.kind {
            ParamKind::Curve => 128.0,
            _ => 23.0,
        };
    }
    [260.0, h]
}

/// Lay out every node of `g` (comments are left where they are).
pub fn auto_layout(g: &mut Graph) {
    let ids: Vec<u32> = g.nodes.iter().filter(|n| n.kind != "comment").map(|n| n.id).collect();
    if ids.is_empty() {
        return;
    }
    // consumers of each node
    let mut outs: HashMap<u32, Vec<u32>> = HashMap::new();
    for l in &g.links {
        outs.entry(l.from_node).or_default().push(l.to_node);
    }
    // rank = longest path to a sink (the output, or the end of a dangling chain)
    let mut rank: HashMap<u32, usize> = HashMap::new();
    fn visit(id: u32, outs: &HashMap<u32, Vec<u32>>, rank: &mut HashMap<u32, usize>, depth: usize) -> usize {
        if let Some(&r) = rank.get(&id) {
            return r;
        }
        if depth > 512 {
            return 0; // cycles can't happen through typed pins, but stay safe
        }
        let r = outs.get(&id).map(|v| v.iter().map(|&t| visit(t, outs, rank, depth + 1) + 1).max().unwrap_or(0)).unwrap_or(0);
        rank.insert(id, r);
        r
    }
    for &id in &ids {
        visit(id, &outs, &mut rank, 0);
    }
    let max_rank = ids.iter().map(|id| rank[id]).max().unwrap_or(0);
    let mut columns: Vec<Vec<u32>> = vec![Vec::new(); max_rank + 1];
    for &id in &ids {
        columns[rank[&id]].push(id);
    }
    // keep the authored vertical order as the starting point
    let pos_y: HashMap<u32, f32> = g.nodes.iter().map(|n| (n.id, n.pos[1])).collect();
    for c in &mut columns {
        c.sort_by(|a, b| pos_y[a].total_cmp(&pos_y[b]));
    }
    // place column by column, from the output leftwards
    let mut y_of: HashMap<u32, f32> = HashMap::new();
    for (r, col) in columns.iter_mut().enumerate() {
        if r > 0 {
            // barycentre of the consumers (already placed: they have a lower rank)
            let key = |id: &u32| {
                let ys: Vec<f32> = outs.get(id).map(|v| v.iter().filter_map(|t| y_of.get(t).copied()).collect()).unwrap_or_default();
                if ys.is_empty() { f32::MAX } else { ys.iter().sum::<f32>() / ys.len() as f32 }
            };
            col.sort_by(|a, b| key(a).total_cmp(&key(b)));
        }
        let mut y = 0.0;
        for id in col.iter() {
            let kind = g.node(*id).map(|n| n.kind.clone()).unwrap_or_default();
            // move down towards the consumers if there is room (keeps links short)
            if r > 0 {
                let ys: Vec<f32> = outs.get(id).map(|v| v.iter().filter_map(|t| y_of.get(t).copied()).collect()).unwrap_or_default();
                if !ys.is_empty() {
                    let want = ys.iter().sum::<f32>() / ys.len() as f32;
                    y = f32::max(y, want);
                }
            }
            y_of.insert(*id, y);
            y += node_size(&kind)[1] + ROW_GAP;
        }
    }
    let x_of = |id: u32| (max_rank - rank[&id]) as f32 * COLUMN_WIDTH;
    for n in &mut g.nodes {
        if let Some(&y) = y_of.get(&n.id) {
            n.pos = [x_of(n.id), y];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::default_project;

    /// Two nodes overlap if their estimated rectangles intersect.
    fn overlaps(g: &Graph) -> usize {
        let rects: Vec<(f32, f32, f32, f32)> = g
            .nodes
            .iter()
            .filter(|n| n.kind != "comment")
            .map(|n| {
                let s = node_size(&n.kind);
                (n.pos[0], n.pos[1], n.pos[0] + s[0], n.pos[1] + s[1])
            })
            .collect();
        let mut count = 0;
        for i in 0..rects.len() {
            for j in i + 1..rects.len() {
                let (a, b) = (rects[i], rects[j]);
                if a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3 {
                    count += 1;
                }
            }
        }
        count
    }

    #[test]
    fn default_graphs_lay_out_without_overlaps() {
        for mut g in default_project().graphs {
            auto_layout(&mut g);
            assert_eq!(overlaps(&g), 0, "graph {} has overlapping nodes", g.name);
            // data flows left to right: every link goes to a column further right
            for l in &g.links {
                let (a, b) = (g.node(l.from_node).unwrap().pos[0], g.node(l.to_node).unwrap().pos[0]);
                assert!(a < b, "link {} -> {} in {} goes backwards", l.from_node, l.to_node, g.name);
            }
        }
    }
}
