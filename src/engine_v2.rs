/// Phase 2: Dual-bitmap shift-and-AND engine.
/// Uses `dist` (distance bitmap) and `ruler` (reverse ruler bitmap)
/// for register-level bit operations instead of per-distance subtraction loops.
use crate::bitmap::Bitmap;

/// Search state for the bitmask engine.
#[derive(Clone, Copy)]
struct State<const W: usize> {
    dist: Bitmap<W>,  // bit d set ⟺ distance d is used
    ruler: Bitmap<W>, // bit r set ⟺ mark at distance r left of rightmost
    pos: u32,         // position of rightmost mark
    depth: usize,     // number of marks placed
    first_gap: u32,   // first gap (for symmetry breaking in later phases)
}

/// Find a Golomb ruler with `n` marks of length <= `max_len`.
pub fn find<const W: usize>(n: usize, max_len: u32) -> Option<Vec<u32>> {
    if n <= 1 {
        return Some(vec![0; n.min(1)]);
    }

    // Initial state: only mark 0 placed
    let mut ruler = Bitmap::<W>::ZERO;
    ruler.set_bit(0); // distance 0 to itself
    let state = State {
        dist: Bitmap::ZERO,
        ruler,
        pos: 0,
        depth: 1,
        first_gap: 0,
    };

    let mut gaps = vec![0u32; n - 1];
    if dfs_find(state, n, max_len, &mut gaps, 0) {
        // Reconstruct marks from gaps
        let mut marks = vec![0u32; n];
        marks[0] = 0;
        for i in 0..n - 1 {
            marks[i + 1] = marks[i] + gaps[i];
        }
        Some(marks)
    } else {
        None
    }
}

/// Dispatch to the right W based on max_len.
pub fn find_dispatched(n: usize, max_len: u32) -> Option<Vec<u32>> {
    dispatch!(n, max_len, find)
}

fn dfs_find<const W: usize>(
    state: State<W>,
    n: usize,
    max_len: u32,
    gaps: &mut [u32],
    gap_idx: usize,
) -> bool {
    if state.depth == n {
        return true;
    }

    let rem = n - state.depth;
    let max_gap = max_len.saturating_sub(state.pos);

    // Each remaining gap must be at least 1
    let gap_ceiling = max_gap.saturating_sub(rem as u32 - 1);

    let mut newbits = Bitmap::<W>::ZERO;

    for gap in 1..=gap_ceiling {
        state.ruler.shl_into(gap as usize, &mut newbits);

        if newbits.intersects(&state.dist) {
            continue; // distance conflict
        }

        // Place mark: update state
        let mut new_state = state;
        new_state.dist |= newbits;
        new_state.ruler = newbits;
        new_state.ruler.set_bit(0);
        new_state.pos += gap;
        new_state.depth = state.depth + 1;
        if state.depth == 1 {
            new_state.first_gap = gap;
        } else {
            new_state.first_gap = state.first_gap;
        }

        gaps[gap_idx] = gap;
        if dfs_find(new_state, n, max_len, gaps, gap_idx + 1) {
            return true;
        }
        // State is on the stack (Copy), so backtracking is automatic
    }

    false
}

/// Check existence exhaustively (for proof mode).
#[allow(dead_code)]
pub fn exists<const W: usize>(n: usize, max_len: u32) -> bool {
    if n <= 1 {
        return true;
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

    dfs_exists(state, n, max_len)
}

#[allow(dead_code)]
pub fn exists_dispatched(n: usize, max_len: u32) -> bool {
    dispatch!(n, max_len, exists)
}

#[allow(dead_code)]
fn dfs_exists<const W: usize>(state: State<W>, n: usize, max_len: u32) -> bool {
    if state.depth == n {
        return true;
    }

    let rem = n - state.depth;
    let max_gap = max_len.saturating_sub(state.pos);
    let gap_ceiling = max_gap.saturating_sub(rem as u32 - 1);

    let mut newbits = Bitmap::<W>::ZERO;

    for gap in 1..=gap_ceiling {
        state.ruler.shl_into(gap as usize, &mut newbits);

        if newbits.intersects(&state.dist) {
            continue;
        }

        let mut new_state = state;
        new_state.dist |= newbits;
        new_state.ruler = newbits;
        new_state.ruler.set_bit(0);
        new_state.pos += gap;
        new_state.depth += 1;
        if state.depth == 1 {
            new_state.first_gap = gap;
        } else {
            new_state.first_gap = state.first_gap;
        }

        if dfs_exists(new_state, n, max_len) {
            return true;
        }
    }

    false
}

/// Macro to dispatch const-generic W based on max_len.
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

use dispatch;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::known::optimal_length;

    #[test]
    fn test_v2_ogr_small() {
        for n in 2..=13 {
            let expected = optimal_length(n).unwrap();
            let result = find::<2>(n, expected).unwrap();
            assert_eq!(
                *result.last().unwrap(),
                expected,
                "V2 OGR-{} should be {}",
                n,
                expected
            );
            crate::naive::verify_golomb(&result);
        }
    }

    #[test]
    fn test_v2_vs_naive() {
        // Cross-validate bitmask engine against naive engine for small n
        for n in 2..=10 {
            for max_len in 1..=optimal_length(n).unwrap() + 2 {
                let naive_result = crate::naive::find(n, max_len).is_some();
                let v2_result = find::<2>(n, max_len).is_some();
                assert_eq!(
                    naive_result, v2_result,
                    "Disagreement at n={}, max_len={}: naive={}, v2={}",
                    n, max_len, naive_result, v2_result
                );
            }
        }
    }

    #[test]
    fn test_v2_ogr_10() {
        let result = find::<1>(10, 55).unwrap();
        assert_eq!(*result.last().unwrap(), 55);
    }
}
