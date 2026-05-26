/// Phase 4+5: Ultimate parallel engine.
/// Combines all optimizations:
///   - Dual-bitmap shift-and-AND (Phase 2)
///   - Branch & bound with static lower bound + node-level dynamic bound
///   - Symmetry breaking (Phase 3)
///   - rayon recursive work-stealing (Phase 4 + Lock ④)
///   - AtomicU32 global best (Phase 4)
///   - local_best caching (Phase 5 Lock ③)
///   - Cache-line aligned shared state (Phase 5 Lock ③)
///   - Branchless cross-word shift in bitmap (Lock ①)

use crate::bitmap::Bitmap;
use crate::known::OGR_OPTIMAL;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// Cache-line aligned atomic best to avoid false sharing.
#[repr(align(64))]
struct AlignedAtomicU32 {
    value: AtomicU32,
}

impl AlignedAtomicU32 {
    fn new(val: u32) -> Self {
        Self {
            value: AtomicU32::new(val),
        }
    }

    #[inline(always)]
    fn load(&self) -> u32 {
        self.value.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn try_update(&self, new_val: u32) -> bool {
        loop {
            let current = self.value.load(Ordering::Relaxed);
            if new_val >= current {
                return false;
            }
            match self.value.compare_exchange_weak(current, new_val, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return true,
                Err(_) => continue,
            }
        }
    }
}

#[derive(Clone, Copy)]
struct State<const W: usize> {
    dist: Bitmap<W>,
    ruler: Bitmap<W>,
    pos: u32,
    depth: usize,
    first_gap: u32,
}

/// Find the shortest Golomb ruler with `n` marks using all threads.
pub fn find_optimal<const W: usize>(n: usize, start_bound: u32, threads: usize) -> Option<(u32, Vec<u32>)> {
    if n <= 1 {
        return Some((0, vec![0; n.min(1)]));
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

    let global_best = Arc::new(AlignedAtomicU32::new(start_bound + 1));
    let best_marks: Arc<std::sync::Mutex<Option<(u32, Vec<u32>)>>> = Arc::new(std::sync::Mutex::new(None));
    let node_count = Arc::new(AtomicU64::new(0));

    pool.install(|| {
        // Generate initial stubs by enumerating first few gaps
        let stubs = generate_stubs::<W>(n, start_bound);

        stubs.into_par_iter().for_each(|(state, gaps)| {
            let local_best = start_bound + 1;
            let mut local_gaps = gaps;
            dfs_parallel(
                state,
                n,
                &global_best,
                local_best,
                &mut local_gaps,
                &best_marks,
                &node_count,
                0,
            );
        });
    });

    let total_nodes = node_count.load(Ordering::Relaxed);
    eprintln!("STAT: total_nodes = {}", total_nodes);

    let result = best_marks.lock().unwrap().take();
    result
}

pub fn find_optimal_dispatched(n: usize, start_bound: u32, threads: usize) -> Option<(u32, Vec<u32>)> {
    let words = crate::known::required_words(start_bound);
    match words {
        1 => find_optimal::<1>(n, start_bound, threads),
        2 => find_optimal::<2>(n, start_bound, threads),
        3 => find_optimal::<3>(n, start_bound, threads),
        4 => find_optimal::<4>(n, start_bound, threads),
        5 => find_optimal::<5>(n, start_bound, threads),
        6 => find_optimal::<6>(n, start_bound, threads),
        7 => find_optimal::<7>(n, start_bound, threads),
        8 => find_optimal::<8>(n, start_bound, threads),
        9 => find_optimal::<9>(n, start_bound, threads),
        10 => find_optimal::<10>(n, start_bound, threads),
        _ => panic!("max_len too large: {}", start_bound),
    }
}

/// Generate stubs: enumerate all valid placements for the first STUB_DEPTH marks.
/// Each stub becomes an independent work unit for parallel processing.
fn generate_stubs<const W: usize>(n: usize, max_len: u32) -> Vec<(State<W>, Vec<u32>)> {
    let stub_depth = if n <= 6 { 2 } else if n <= 12 { 3 } else { 4 };
    let mut stubs = Vec::new();

    let mut ruler = Bitmap::<W>::ZERO;
    ruler.set_bit(0);
    let initial = State {
        dist: Bitmap::ZERO,
        ruler,
        pos: 0,
        depth: 1,
        first_gap: 0,
    };

    let mut gaps = vec![0u32; n - 1];
    enumerate_stubs(initial, n, max_len, stub_depth, &mut gaps, 0, &mut stubs);

    if stubs.is_empty() {
        // Fallback: use initial state as single stub
        stubs.push((initial, vec![0u32; n - 1]));
    }

    stubs
}

fn enumerate_stubs<const W: usize>(
    state: State<W>,
    n: usize,
    max_len: u32,
    target_depth: usize,
    gaps: &mut [u32],
    gap_idx: usize,
    stubs: &mut Vec<(State<W>, Vec<u32>)>,
) {
    if state.depth == target_depth || state.depth == n {
        stubs.push((state, gaps.to_vec()));
        return;
    }

    let rem = n - state.depth;
    let max_gap = max_len - state.pos;
    let gap_ceiling = max_gap.saturating_sub(rem as u32 - 1);
    if gap_ceiling == 0 {
        return;
    }

    let mut newbits = Bitmap::<W>::ZERO;

    for gap in 1..=gap_ceiling {
        if state.depth == n - 1 && gap as u32 <= state.first_gap {
            continue;
        }

        state.ruler.shl_into(gap as usize, &mut newbits);
        if newbits.intersects(&state.dist) {
            continue;
        }

        // Static lower bound check
        if rem + 1 < OGR_OPTIMAL.len() {
            let new_pos = state.pos + gap as u32;
            let bound = new_pos + OGR_OPTIMAL[rem];
            if bound > max_len {
                continue;
            }
        }

        let mut new_state = state;
        new_state.dist |= newbits;
        new_state.ruler = newbits;
        new_state.ruler.set_bit(0);
        new_state.pos += gap as u32;
        new_state.depth += 1;
        if state.depth == 1 {
            new_state.first_gap = gap as u32;
        }

        gaps[gap_idx] = gap as u32;
        enumerate_stubs(new_state, n, max_len, target_depth, gaps, gap_idx + 1, stubs);
    }
}

/// Parallel DFS with local_best caching (Lock ③) and recursive work-stealing (Lock ④).
const SYNC_INTERVAL: u32 = 50_000;
const PARALLEL_GRAIN_DEPTH: usize = 6;

fn dfs_parallel<const W: usize>(
    state: State<W>,
    n: usize,
    global_best: &AlignedAtomicU32,
    mut local_best: u32,
    gaps: &mut [u32],
    best_marks: &Arc<std::sync::Mutex<Option<(u32, Vec<u32>)>>>,
    node_count: &AtomicU64,
    mut iter_count: u32,
) {
    node_count.fetch_add(1, Ordering::Relaxed);
    if state.depth == n {
        if state.pos < local_best {
            let mut marks = vec![0u32; n];
            marks[0] = 0;
            for i in 0..n - 1 {
                marks[i + 1] = marks[i] + gaps[i];
            }

            let mut guard = best_marks.lock().unwrap();
            let current_best = guard.as_ref().map(|(l, _)| *l).unwrap_or(u32::MAX);
            if state.pos < current_best {
                *guard = Some((state.pos, marks));
                global_best.try_update(state.pos);
            }
        }
        return;
    }

    let rem = n - state.depth;

    // Periodic sync: read global best into local (Lock ③)
    iter_count += 1;
    if iter_count >= SYNC_INTERVAL {
        iter_count = 0;
        let global = global_best.load();
        if global < local_best {
            local_best = global;
        }
    }

    // Branch & bound: static lower bound
    if rem + 1 < OGR_OPTIMAL.len() {
        let static_bound = state.pos + OGR_OPTIMAL[rem + 1];
        if static_bound >= local_best {
            return;
        }
    }

    // Per-node dynamic bound: sum of rem smallest available distances
    if rem >= 1 {
        match state.dist.sum_smallest_unset(rem, local_best as usize) {
            Some(s) if state.pos + s < local_best => {}
            _ => return,
        }
    }

    let max_gap = local_best.saturating_sub(state.pos);
    if max_gap == 0 {
        return;
    }
    let gap_ceiling = max_gap.saturating_sub(rem as u32 - 1);
    if gap_ceiling == 0 {
        return;
    }

    // Lock ④: recursive parallelism with grain-size threshold
    if rem > PARALLEL_GRAIN_DEPTH {
        // Parallel path: use rayon::join for binary splitting
        dfs_parallel_recursive::<W>(
            state, n, global_best, local_best, gaps, best_marks, node_count, iter_count, gap_ceiling,
        );
    } else {
        // Serial path: standard DFS
        dfs_serial::<W>(state, n, &mut local_best, global_best, gaps, best_marks, node_count, iter_count);
    }
}

const MAX_INLINE_GAPS: usize = 128;

fn dfs_parallel_recursive<const W: usize>(
    state: State<W>,
    n: usize,
    global_best: &AlignedAtomicU32,
    local_best: u32,
    gaps: &mut [u32],
    best_marks: &Arc<std::sync::Mutex<Option<(u32, Vec<u32>)>>>,
    node_count: &AtomicU64,
    iter_count: u32,
    gap_ceiling: u32,
) {
    let rem = n - state.depth;

    // Collect valid gaps. Only store the gap value (not the bitmap — that's
    // W-dependent and large). Recompute the shift in the child closure; one
    // O(W) shift per child is negligible against the subtree it explores.
    let mut newbits = state.ruler.shl(1);

    let mut valid_gaps: SmallGapBuf = SmallGapBuf::new(gap_ceiling as usize);

    for gap in 1..=gap_ceiling {
        if gap > 1 {
            newbits.shl_one();
        }
        if state.depth == n - 1 && gap <= state.first_gap {
            continue;
        }
        if newbits.intersects(&state.dist) {
            continue;
        }
        if rem < OGR_OPTIMAL.len() {
            let new_pos = state.pos + gap;
            if new_pos + OGR_OPTIMAL[rem] >= local_best {
                continue;
            }
        }

        valid_gaps.push(gap);
    }

    if valid_gaps.is_empty() {
        return;
    }

    let make_child = |gap: u32| -> State<W> {
        let nb = state.ruler.shl(gap as usize);
        let mut new_state = state;
        new_state.dist |= nb;
        new_state.ruler = nb;
        new_state.ruler.set_bit(0);
        new_state.pos += gap;
        new_state.depth += 1;
        if state.depth == 1 {
            new_state.first_gap = gap;
        }
        new_state
    };

    if valid_gaps.len() == 1 {
        let gap = valid_gaps.get(0);
        let new_state = make_child(gap);
        gaps[state.depth - 1] = gap;
        dfs_parallel(new_state, n, global_best, local_best, gaps, best_marks, node_count, iter_count);
        return;
    }

    valid_gaps.as_slice().par_iter().for_each(|&gap| {
        let new_state = make_child(gap);
        let mut local_gaps = [0u32; 64];
        local_gaps[..state.depth - 1].copy_from_slice(&gaps[..state.depth - 1]);
        local_gaps[state.depth - 1] = gap;

        dfs_parallel(new_state, n, global_best, local_best, &mut local_gaps[..n - 1], best_marks, node_count, 0);
    });
}

/// Stack-allocated gap buffer that falls back to Vec for large gap ceilings.
struct SmallGapBuf {
    inline: [u32; MAX_INLINE_GAPS],
    heap: Option<Vec<u32>>,
    len: usize,
}

impl SmallGapBuf {
    fn new(capacity: usize) -> Self {
        Self {
            inline: [0; MAX_INLINE_GAPS],
            heap: if capacity > MAX_INLINE_GAPS { Some(Vec::with_capacity(capacity)) } else { None },
            len: 0,
        }
    }

    fn push(&mut self, val: u32) {
        if let Some(ref mut heap) = self.heap {
            heap.push(val);
        } else {
            debug_assert!(self.len < MAX_INLINE_GAPS);
            self.inline[self.len] = val;
        }
        self.len += 1;
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_slice(&self) -> &[u32] {
        if let Some(ref heap) = self.heap {
            heap.as_slice()
        } else {
            &self.inline[..self.len]
        }
    }

    fn get(&self, idx: usize) -> u32 {
        if let Some(ref heap) = self.heap {
            heap[idx]
        } else {
            self.inline[idx]
        }
    }
}

fn dfs_serial<const W: usize>(
    state: State<W>,
    n: usize,
    local_best: &mut u32,
    global_best: &AlignedAtomicU32,
    gaps: &mut [u32],
    best_marks: &Arc<std::sync::Mutex<Option<(u32, Vec<u32>)>>>,
    node_count: &AtomicU64,
    mut iter_count: u32,
) {
    node_count.fetch_add(1, Ordering::Relaxed);
    if state.depth == n {
        if state.pos < *local_best {
            *local_best = state.pos;

            let mut marks = vec![0u32; n];
            marks[0] = 0;
            for i in 0..n - 1 {
                marks[i + 1] = marks[i] + gaps[i];
            }

            let mut guard = best_marks.lock().unwrap();
            let current_best = guard.as_ref().map(|(l, _)| *l).unwrap_or(u32::MAX);
            if state.pos < current_best {
                *guard = Some((state.pos, marks));
                global_best.try_update(state.pos);
            }
        }
        return;
    }

    let rem = n - state.depth;

    // Periodic sync (Lock ③)
    iter_count += 1;
    if iter_count >= SYNC_INTERVAL {
        iter_count = 0;
        let global = global_best.load();
        if global < *local_best {
            *local_best = global;
        }
    }

    // Branch & bound: static lower bound
    if rem + 1 < OGR_OPTIMAL.len() {
        let static_bound = state.pos + OGR_OPTIMAL[rem + 1];
        if static_bound >= *local_best {
            return;
        }
    }

    // Per-node dynamic bound: sum of rem smallest available distances
    if rem >= 1 {
        match state.dist.sum_smallest_unset(rem, *local_best as usize) {
            Some(s) if state.pos + s < *local_best => {}
            _ => return,
        }
    }

    let max_gap = local_best.saturating_sub(state.pos);
    if max_gap == 0 {
        return;
    }
    let gap_ceiling = max_gap.saturating_sub(rem as u32 - 1);
    if gap_ceiling == 0 {
        return;
    }

    let mut newbits = state.ruler.shl(1);

    for gap in 1..=gap_ceiling {
        if gap > 1 {
            newbits.shl_one();
        }
        if state.depth == n - 1 && gap <= state.first_gap {
            continue;
        }

        if newbits.intersects(&state.dist) {
            continue;
        }

        // Per-gap static bound
        if rem < OGR_OPTIMAL.len() {
            let new_pos = state.pos + gap;
            if new_pos + OGR_OPTIMAL[rem] >= *local_best {
                continue;
            }
        }

        let mut new_state = state;
        new_state.dist |= newbits;
        new_state.ruler = newbits;
        new_state.ruler.set_bit(0);
        new_state.pos += gap;
        new_state.depth += 1;
        if state.depth == 1 {
            new_state.first_gap = gap;
        }

        gaps[state.depth - 1] = gap;
        dfs_serial(new_state, n, local_best, global_best, gaps, best_marks, node_count, iter_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::known::optimal_length;

    #[test]
    fn test_v4_ogr_up_to_13() {
        for n in 2..=13 {
            let expected = optimal_length(n).unwrap();
            let (len, marks) = find_optimal::<2>(n, expected + 5, 2).unwrap();
            assert_eq!(len, expected, "V4 OGR-{} should be {}, got {}", n, expected, len);
            assert_eq!(*marks.last().unwrap(), len, "V4 OGR-{} ruler length mismatch", n);
            crate::naive::verify_golomb(&marks);
        }
    }

    #[test]
    #[ignore] // Slow: run with `cargo test --release -- --ignored`
    fn test_v4_ogr_up_to_18() {
        for n in 14..=18 {
            let expected = optimal_length(n).unwrap();
            let (len, marks) = find_optimal_dispatched(n, expected + 5, 4).unwrap();
            assert_eq!(len, expected, "V4 OGR-{} should be {}, got {}", n, expected, len);
            assert_eq!(*marks.last().unwrap(), len, "V4 OGR-{} ruler length mismatch", n);
            crate::naive::verify_golomb(&marks);
        }
    }

    #[test]
    fn test_v4_ogr_10() {
        let (len, marks) = find_optimal::<1>(10, 55, 2).unwrap();
        assert_eq!(len, 55);
        assert_eq!(*marks.last().unwrap(), 55);
        crate::naive::verify_golomb(&marks);
    }

}
