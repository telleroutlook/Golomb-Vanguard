/// Phase 3: Branch & bound engine with symmetry breaking and dual lower bounds.
/// Adds:
///   - Symmetry breaking: g_first < g_last (halves search space)
///   - Static lower bound: P + OGR_MIN[rem+1] >= best → prune
///   - Dynamic lower bound (Phase 5 Lock ②): sum of smallest unused distances
///   - Incremental available-distance cache (Lock ② advanced)
/// Both bounds are combined via max().
use crate::avail::AvailDistances;
use crate::bitmap::Bitmap;
use crate::known::OGR_OPTIMAL;

#[derive(Clone, Copy)]
struct State<const W: usize> {
    dist: Bitmap<W>,
    ruler: Bitmap<W>,
    pos: u32,
    depth: usize,
    first_gap: u32,
}

/// Find the shortest Golomb ruler with `n` marks.
/// Searches from a starting upper bound downward.
/// Returns (length, marks) of the optimal ruler.
pub fn find_optimal<const W: usize>(n: usize, start_bound: u32) -> Option<(u32, Vec<u32>)> {
    if n <= 1 {
        return Some((0, vec![0; n.min(1)]));
    }

    let mut best = start_bound + 1;
    let mut best_marks: Option<Vec<u32>> = None;

    let mut ruler = Bitmap::<W>::ZERO;
    ruler.set_bit(0);
    let state = State {
        dist: Bitmap::ZERO,
        ruler,
        pos: 0,
        depth: 1,
        first_gap: 0,
    };

    let mut gaps = vec![0u32; n - 1];
    dfs_search(state, n, &mut best, &mut gaps, &mut best_marks);

    best_marks.map(|m| (best, m))
}

pub fn find_optimal_dispatched(n: usize, start_bound: u32) -> Option<(u32, Vec<u32>)> {
    dispatch!(n, start_bound, find_optimal)
}

/// Search for any Golomb ruler with `n` marks of length <= `max_len`.
pub fn find<const W: usize>(n: usize, max_len: u32) -> Option<Vec<u32>> {
    find_optimal::<W>(n, max_len).map(|(_, m)| m)
}

pub fn find_dispatched(n: usize, max_len: u32) -> Option<Vec<u32>> {
    find_optimal_dispatched(n, max_len).map(|(_, m)| m)
}

/// Prove no Golomb ruler with `n` marks of length <= `max_len` exists.
/// Returns true if proven impossible (exhaustive search found nothing).
pub fn prove_impossible<const W: usize>(n: usize, max_len: u32) -> bool {
    if n <= 1 {
        return false;
    }
    let mut ruler = Bitmap::<W>::ZERO;
    ruler.set_bit(0);
    let state = State {
        dist: Bitmap::ZERO,
        ruler,
        pos: 0,
        depth: 1,
        first_gap: 0,
    };
    !dfs_exists(state, n, max_len)
}

pub fn prove_impossible_dispatched(n: usize, max_len: u32) -> bool {
    dispatch_bool!(n, max_len, prove_impossible)
}

fn dfs_search<const W: usize>(
    state: State<W>,
    n: usize,
    best: &mut u32,
    gaps: &mut [u32],
    best_marks: &mut Option<Vec<u32>>,
) {
    if state.depth == n {
        if state.pos < *best {
            *best = state.pos;
            let mut marks = vec![0u32; n];
            marks[0] = 0;
            for i in 0..n - 1 {
                marks[i + 1] = marks[i] + gaps[i];
            }
            *best_marks = Some(marks);
        }
        return;
    }

    let rem = n - state.depth;
    let max_gap = *best - state.pos;
    if max_gap == 0 {
        return;
    }

    // Branch & bound: static lower bound
    if rem + 1 < OGR_OPTIMAL.len() {
        let static_bound = state.pos + OGR_OPTIMAL[rem + 1];
        if static_bound >= *best {
            return;
        }
    }

    let gap_ceiling = max_gap.saturating_sub(rem as u32 - 1);
    if gap_ceiling == 0 {
        return;
    }

    // Precompute available distances for dynamic bound (incremental Lock ②)
    let base_avail = if rem >= 2 {
        Some(AvailDistances::from_bitmap(
            &state.dist,
            rem - 1,
            *best as usize,
        ))
    } else {
        None
    };

    let mut newbits = Bitmap::<W>::ZERO;

    for gap in 1..=gap_ceiling {
        if state.depth == n - 1 && gap as u32 <= state.first_gap {
            continue;
        }

        state.ruler.shl_into(gap as usize, &mut newbits);
        if newbits.intersects(&state.dist) {
            continue;
        }

        // Incremental dynamic lower bound
        if let Some(ref base) = base_avail {
            let new_pos = state.pos + gap as u32;
            let filtered = AvailDistances::without_bitmap::<W>(base, &newbits);
            if let Some(dynamic_sum) = filtered.sum_k(rem - 1) {
                let dynamic_bound = new_pos + dynamic_sum;
                if rem + 1 < OGR_OPTIMAL.len() {
                    let static_bound = new_pos + OGR_OPTIMAL[rem];
                    if dynamic_bound.max(static_bound) >= *best {
                        continue;
                    }
                } else if dynamic_bound >= *best {
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

        gaps[state.depth - 1] = gap as u32;
        dfs_search(new_state, n, best, gaps, best_marks);
    }
}

fn dfs_exists<const W: usize>(state: State<W>, n: usize, max_len: u32) -> bool {
    if state.depth == n {
        return true;
    }

    let rem = n - state.depth;
    let max_gap = max_len - state.pos;
    let gap_ceiling = max_gap.saturating_sub(rem as u32 - 1);
    if gap_ceiling == 0 {
        return false;
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

        let mut new_state = state;
        new_state.dist |= newbits;
        new_state.ruler = newbits;
        new_state.ruler.set_bit(0);
        new_state.pos += gap as u32;
        new_state.depth += 1;
        if state.depth == 1 {
            new_state.first_gap = gap as u32;
        }

        if dfs_exists(new_state, n, max_len) {
            return true;
        }
    }

    false
}

macro_rules! dispatch {
    ($n:expr, $max_len:expr, $func:ident) => {{
        let n = $n;
        let ml = $max_len;
        let words = crate::known::required_words(ml);
        match words {
            1 => $func::<1>(n, ml),
            2 => $func::<2>(n, ml),
            3 => $func::<3>(n, ml),
            4 => $func::<4>(n, ml),
            5 => $func::<5>(n, ml),
            6 => $func::<6>(n, ml),
            7 => $func::<7>(n, ml),
            8 => $func::<8>(n, ml),
            9 => $func::<9>(n, ml),
            10 => $func::<10>(n, ml),
            _ => panic!("max_len too large: {}", ml),
        }
    }};
}

macro_rules! dispatch_bool {
    ($n:expr, $max_len:expr, $func:ident) => {{
        let n = $n;
        let ml = $max_len;
        let words = crate::known::required_words(ml);
        match words {
            1 => $func::<1>(n, ml),
            2 => $func::<2>(n, ml),
            3 => $func::<3>(n, ml),
            4 => $func::<4>(n, ml),
            5 => $func::<5>(n, ml),
            6 => $func::<6>(n, ml),
            7 => $func::<7>(n, ml),
            8 => $func::<8>(n, ml),
            9 => $func::<9>(n, ml),
            10 => $func::<10>(n, ml),
            _ => panic!("max_len too large: {}", ml),
        }
    }};
}

use dispatch;
use dispatch_bool;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::known::optimal_length;

    #[test]
    fn test_v3_ogr_up_to_13() {
        for n in 2..=13 {
            let expected = optimal_length(n).unwrap();
            let (len, marks) = find_optimal::<2>(n, expected + 5).unwrap();
            assert_eq!(
                len, expected,
                "V3 OGR-{} should be {}, got {}",
                n, expected, len
            );
            crate::naive::verify_golomb(&marks);
        }
    }

    #[test]
    #[ignore] // Slow: run with `cargo test --release -- --ignored`
    fn test_v3_ogr_up_to_18() {
        for n in 14..=18 {
            let expected = optimal_length(n).unwrap();
            let words = crate::known::required_words(expected);
            let (len, marks) = find_optimal_dispatched(n, expected + 5).unwrap();
            assert_eq!(
                len, expected,
                "V3 OGR-{} should be {}, got {}",
                n, expected, len
            );
            crate::naive::verify_golomb(&marks);
        }
    }

    #[test]
    fn test_v3_symmetry_breaking() {
        // Ensure symmetry breaking doesn't miss solutions
        for n in 2..=10 {
            let naive_result = crate::naive::find(n, optimal_length(n).unwrap());
            let v3_result = find::<2>(n, optimal_length(n).unwrap());
            assert_eq!(
                naive_result.is_some(),
                v3_result.is_some(),
                "V3 symmetry breaking broke OGR-{}",
                n
            );
        }
    }

    #[test]
    fn test_v3_prove_impossible() {
        assert!(prove_impossible::<2>(6, 16));
        assert!(!prove_impossible::<2>(6, 17));
    }
}
