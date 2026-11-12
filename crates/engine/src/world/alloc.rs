//! First-fit range allocator with coalescing (quad pool sub-allocation).

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub offset: u32,
    pub size: u32,
}

pub struct RangeAllocator {
    /// offset -> size of free blocks
    free: BTreeMap<u32, u32>,
    pub capacity: u32,
    pub used: u64,
}

impl RangeAllocator {
    pub fn new(capacity: u32) -> Self {
        let mut free = BTreeMap::new();
        free.insert(0, capacity);
        Self { free, capacity, used: 0 }
    }

    pub fn alloc(&mut self, size: u32) -> Option<Range> {
        if size == 0 {
            return Some(Range { offset: 0, size: 0 });
        }
        // Round to 64 (one meshlet) to limit fragmentation.
        let size = size.div_ceil(64) * 64;
        let (&off, &bsize) = self.free.iter().find(|&(_, &s)| s >= size)?;
        self.free.remove(&off);
        if bsize > size {
            self.free.insert(off + size, bsize - size);
        }
        self.used += size as u64;
        Some(Range { offset: off, size })
    }

    pub fn free(&mut self, r: Range) {
        if r.size == 0 {
            return;
        }
        self.used -= r.size as u64;
        let mut off = r.offset;
        let mut size = r.size;
        // merge with previous
        if let Some((&poff, &psize)) = self.free.range(..off).next_back() {
            if poff + psize == off {
                self.free.remove(&poff);
                off = poff;
                size += psize;
            }
        }
        // merge with next
        if let Some(&nsize) = self.free.get(&(off + size)) {
            self.free.remove(&(off + size));
            size += nsize;
        }
        self.free.insert(off, size);
    }

    pub fn largest_free(&self) -> u32 {
        self.free.values().copied().max().unwrap_or(0)
    }

    pub fn fragments(&self) -> usize {
        self.free.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn alloc_free_coalesce() {
        let mut a = RangeAllocator::new(1024);
        let x = a.alloc(100).unwrap();
        let y = a.alloc(100).unwrap();
        let z = a.alloc(100).unwrap();
        a.free(y);
        a.free(x);
        a.free(z);
        assert_eq!(a.fragments(), 1);
        assert_eq!(a.largest_free(), 1024);
        assert_eq!(a.used, 0);
    }
}
