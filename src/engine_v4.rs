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
///   - Per-thread node counter (avoids cache-line bouncing)
///   - Dead-end measurement counters by depth
use crate::bitmap::Bitmap;
use crate::known::OGR_OPTIMAL;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
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
            match self.value.compare_exchange_weak(
                current,
                new_val,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
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

#[cfg(feature = "stats")]
use std::cell::RefCell;

#[cfg(feature = "stats")]
struct ThreadStats {
    nodes: u64,
    dead_ends: [u64; 32],
}

#[cfg(feature = "stats")]
impl ThreadStats {
    fn new() -> Self {
        Self {
            nodes: 0,
            dead_ends: [0; 32],
        }
    }
}

#[cfg(feature = "stats")]
thread_local! {
    static THREAD_STATS: RefCell<ThreadStats> = RefCell::new(ThreadStats::new());
}

macro_rules! stat_inc_nodes {
    () => {
        #[cfg(feature = "stats")]
        THREAD_STATS.with(|ts| ts.borrow_mut().nodes += 1);
    };
}

macro_rules! stat_inc_dead_end {
    ($depth:expr) => {
        #[cfg(feature = "stats")]
        {
            let d = $depth;
            if d < 32 {
                THREAD_STATS.with(|ts| ts.borrow_mut().dead_ends[d] += 1);
            }
        }
    };
}

/// Find the shortest Golomb ruler with `n` marks using all threads.
pub fn find_optimal<const W: usize>(
    n: usize,
    start_bound: u32,
    threads: usize,
) -> Option<(u32, Vec<u32>)> {
    if n <= 1 {
        return Some((0, vec![0; n.min(1)]));
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());

    let global_best = Arc::new(AlignedAtomicU32::new(start_bound + 1));
    let best_marks: Arc<std::sync::Mutex<Option<(u32, Vec<u32>)>>> =
        Arc::new(std::sync::Mutex::new(None));

    #[cfg(feature = "stats")]
    let agg_nodes: Arc<std::sync::atomic::AtomicU64> =
        Arc::new(std::sync::atomic::AtomicU64::new(0));
    #[cfg(feature = "stats")]
    let agg_dead_ends: Arc<std::sync::Mutex<[u64; 32]>> =
        Arc::new(std::sync::Mutex::new([0u64; 32]));

    pool.install(|| {
        let stubs = generate_stubs::<W>(n, start_bound);

        stubs.into_par_iter().for_each(|(state, gaps)| {
            #[cfg(feature = "stats")]
            THREAD_STATS.with(|ts| {
                *ts.borrow_mut() = ThreadStats::new();
            });

            let local_best = start_bound + 1;
            let mut local_gaps = gaps;
            dfs_parallel(
                state,
                n,
                &global_best,
                local_best,
                &mut local_gaps,
                &best_marks,
                0,
            );

            #[cfg(feature = "stats")]
            THREAD_STATS.with(|ts| {
                let ts = ts.borrow();
                agg_nodes.fetch_add(ts.nodes, Ordering::Relaxed);
                let mut guard = agg_dead_ends.lock().unwrap();
                for i in 0..32 {
                    guard[i] += ts.dead_ends[i];
                }
            });
        });
    });

    #[cfg(feature = "stats")]
    {
        let total_nodes = agg_nodes.load(Ordering::Relaxed);
        eprintln!("STAT: total_nodes = {}", total_nodes);
        let dead_ends = *agg_dead_ends.lock().unwrap();
        eprintln!("STAT: dead_ends_by_depth:");
        for d in 1..n {
            if dead_ends[d] > 0 {
                eprintln!("  depth {:2}: {}", d, dead_ends[d]);
            }
        }
    }

    let result = best_marks.lock().unwrap().take();
    result
}

pub fn find_optimal_dispatched(
    n: usize,
    start_bound: u32,
    threads: usize,
) -> Option<(u32, Vec<u32>)> {
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
    let stub_depth = if n <= 6 {
        2
    } else if n <= 12 {
        3
    } else {
        4
    };
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
        stubs.push((initial, vec![0u32; n - 1]));
    } else {
        // Heuristic: explore tightest-packing stubs first (smallest pos)
        // so parallel threads find the global best early
        stubs.sort_unstable_by_key(|(state, _)| state.pos);
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

        if rem + 1 < OGR_OPTIMAL.len() {
            let new_pos = state.pos + gap as u32;
            let bound = new_pos + OGR_OPTIMAL[rem];
            if bound > max_len {
                continue;
            }
        }

        // Per-gap symmetry-aware static bound
        if rem >= 2 && (rem - 1) < OGR_OPTIMAL.len() {
            let child_first_gap = if state.depth == 1 {
                gap as u32
            } else {
                state.first_gap
            };
            if child_first_gap > 0 {
                let sym_bound =
                    (state.pos + gap as u32) + OGR_OPTIMAL[rem - 1] + child_first_gap + 1;
                if sym_bound > max_len {
                    continue;
                }
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
        enumerate_stubs(
            new_state,
            n,
            max_len,
            target_depth,
            gaps,
            gap_idx + 1,
            stubs,
        );
    }
}

/// Parallel DFS with local_best caching (Lock ③) and recursive work-stealing (Lock ④).
const SYNC_INTERVAL: u32 = 50_000;
const PARALLEL_GRAIN_DEPTH: usize = 7;

fn dfs_parallel<const W: usize>(
    state: State<W>,
    n: usize,
    global_best: &AlignedAtomicU32,
    mut local_best: u32,
    gaps: &mut [u32],
    best_marks: &Arc<std::sync::Mutex<Option<(u32, Vec<u32>)>>>,
    mut iter_count: u32,
) {
    stat_inc_nodes!();

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

    // Symmetry-aware static bound: OGR_OPTIMAL[rem] + first_gap + 1
    // At least one remaining gap (g_{n-1}) must exceed first_gap
    if state.first_gap > 0 && rem < OGR_OPTIMAL.len() && rem >= 2 {
        let sym_bound = state.pos + OGR_OPTIMAL[rem] + state.first_gap + 1;
        if sym_bound >= local_best {
            return;
        }
    }

    // Per-node dynamic bound: symmetry-aware sum of rem smallest available distances
    if rem >= 1 {
        match state
            .dist
            .sum_smallest_unset_sym(rem, local_best as usize, state.first_gap)
        {
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
        dfs_parallel_recursive::<W>(
            state,
            n,
            global_best,
            local_best,
            gaps,
            best_marks,
            iter_count,
            gap_ceiling,
        );
    } else {
        dfs_serial::<W>(
            state,
            n,
            &mut local_best,
            global_best,
            gaps,
            best_marks,
            iter_count,
        );
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
    iter_count: u32,
    gap_ceiling: u32,
) {
    let rem = n - state.depth;

    // Hoisted per-gap static bound: gap + OGR_OPTIMAL[rem] < local_best - pos
    let max_gap = local_best.saturating_sub(state.pos);
    let mut tight_ceiling = gap_ceiling;
    if rem < OGR_OPTIMAL.len() {
        let static_ceil = max_gap.saturating_sub(OGR_OPTIMAL[rem]).saturating_sub(1);
        if static_ceil < tight_ceiling {
            tight_ceiling = static_ceil;
        }
    }

    // Hoisted per-gap symmetry-aware static bound
    if rem >= 2 && (rem - 1) < OGR_OPTIMAL.len() {
        if state.depth == 1 {
            let sym_ceil = max_gap
                .saturating_sub(OGR_OPTIMAL[rem - 1])
                .saturating_sub(2)
                / 2;
            if sym_ceil < tight_ceiling {
                tight_ceiling = sym_ceil;
            }
        } else if state.first_gap > 0 {
            let sym_ceil = max_gap
                .saturating_sub(OGR_OPTIMAL[rem - 1])
                .saturating_sub(state.first_gap)
                .saturating_sub(2);
            if sym_ceil < tight_ceiling {
                tight_ceiling = sym_ceil;
            }
        }
    }

    // Bitwise parallel gap pre-filtering:
    // forbidden_gaps = union over marks k in ruler: (dist >> k)
    // valid gaps = ~forbidden_gaps, iterated via trailing_zeros
    let mut forbidden = Bitmap::<W>::ZERO;
    for k in state.ruler.iter_set_bits() {
        forbidden |= state.dist.shr(k as usize);
    }
    let mut valid = !forbidden;
    valid.clear_bit(0); // gap 0 is never valid

    let mut valid_gaps: SmallGapBuf = SmallGapBuf::new(tight_ceiling as usize);

    for gap in valid.iter_set_bits() {
        if gap > tight_ceiling {
            break;
        }
        if state.depth == n - 1 && gap <= state.first_gap {
            continue;
        }

        valid_gaps.push(gap);
    }

    if valid_gaps.is_empty() {
        stat_inc_nodes!();
        stat_inc_dead_end!(state.depth);
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
        stat_inc_nodes!();
        let gap = valid_gaps.get(0);
        let new_state = make_child(gap);
        gaps[state.depth - 1] = gap;
        dfs_parallel(
            new_state,
            n,
            global_best,
            local_best,
            gaps,
            best_marks,
            iter_count,
        );
        return;
    }

    stat_inc_nodes!();

    valid_gaps.as_slice().par_iter().for_each(|&gap| {
        let new_state = make_child(gap);
        let mut local_gaps = [0u32; 64];
        local_gaps[..state.depth - 1].copy_from_slice(&gaps[..state.depth - 1]);
        local_gaps[state.depth - 1] = gap;

        dfs_parallel(
            new_state,
            n,
            global_best,
            local_best,
            &mut local_gaps[..n - 1],
            best_marks,
            0,
        );
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
            heap: if capacity > MAX_INLINE_GAPS {
                Some(Vec::with_capacity(capacity))
            } else {
                None
            },
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
    mut iter_count: u32,
) {
    stat_inc_nodes!();

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

    // Symmetry-aware static bound
    if state.first_gap > 0 && rem < OGR_OPTIMAL.len() && rem >= 2 {
        let sym_bound = state.pos + OGR_OPTIMAL[rem] + state.first_gap + 1;
        if sym_bound >= *local_best {
            return;
        }
    }

    // Per-node dynamic bound: symmetry-aware
    if rem >= 1 {
        match state
            .dist
            .sum_smallest_unset_sym(rem, *local_best as usize, state.first_gap)
        {
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
        stat_inc_dead_end!(state.depth);
        return;
    }

    let mut had_child = false;

    if gap_ceiling > state.depth as u32 + 2 {
        // Bitwise pre-filtering path: cheaper when gap_ceiling >> depth
        let mut forbidden = Bitmap::<W>::ZERO;
        for k in state.ruler.iter_set_bits() {
            forbidden |= state.dist.shr(k as usize);
        }
        let mut valid = !forbidden;
        valid.clear_bit(0);

        for gap in valid.iter_set_bits() {
            if gap > gap_ceiling {
                break;
            }
            if state.depth == n - 1 && gap <= state.first_gap {
                continue;
            }

            // Per-gap static bound (local_best may have changed)
            if rem < OGR_OPTIMAL.len() {
                if state.pos + gap + OGR_OPTIMAL[rem] >= *local_best {
                    continue;
                }
            }

            // Per-gap symmetry-aware static bound
            if rem >= 2 && (rem - 1) < OGR_OPTIMAL.len() {
                let child_first_gap = if state.depth == 1 {
                    gap
                } else {
                    state.first_gap
                };
                if child_first_gap > 0 {
                    let sym_bound = (state.pos + gap) + OGR_OPTIMAL[rem - 1] + child_first_gap + 1;
                    if sym_bound >= *local_best {
                        continue;
                    }
                }
            }

            let newbits = state.ruler.shl(gap as usize);
            let mut new_state = state;
            new_state.dist |= newbits;
            new_state.ruler = newbits;
            new_state.ruler.set_bit(0);
            new_state.pos += gap;
            new_state.depth += 1;
            if state.depth == 1 {
                new_state.first_gap = gap;
            }

            had_child = true;
            gaps[state.depth - 1] = gap;
            dfs_serial(
                new_state,
                n,
                local_best,
                global_best,
                gaps,
                best_marks,
                iter_count,
            );
        }
    } else {
        // Incremental shl_one path: cheaper at deep depths
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

            // Per-gap symmetry-aware static bound
            if rem >= 2 && (rem - 1) < OGR_OPTIMAL.len() {
                let child_first_gap = if state.depth == 1 {
                    gap
                } else {
                    state.first_gap
                };
                if child_first_gap > 0 {
                    let sym_bound = (state.pos + gap) + OGR_OPTIMAL[rem - 1] + child_first_gap + 1;
                    if sym_bound >= *local_best {
                        continue;
                    }
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

            had_child = true;
            gaps[state.depth - 1] = gap;
            dfs_serial(
                new_state,
                n,
                local_best,
                global_best,
                gaps,
                best_marks,
                iter_count,
            );
        }
    }

    if !had_child {
        stat_inc_dead_end!(state.depth);
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
            assert_eq!(
                len, expected,
                "V4 OGR-{} should be {}, got {}",
                n, expected, len
            );
            assert_eq!(
                *marks.last().unwrap(),
                len,
                "V4 OGR-{} ruler length mismatch",
                n
            );
            crate::naive::verify_golomb(&marks);
        }
    }

    #[test]
    #[ignore] // Slow: run with `cargo test --release -- --ignored`
    fn test_v4_ogr_up_to_18() {
        for n in 14..=18 {
            let expected = optimal_length(n).unwrap();
            let (len, marks) = find_optimal_dispatched(n, expected + 5, 4).unwrap();
            assert_eq!(
                len, expected,
                "V4 OGR-{} should be {}, got {}",
                n, expected, len
            );
            assert_eq!(
                *marks.last().unwrap(),
                len,
                "V4 OGR-{} ruler length mismatch",
                n
            );
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
