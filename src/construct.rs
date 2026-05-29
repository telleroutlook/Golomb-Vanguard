/// Constructive Golomb ruler generation for seeding find-mode upper bound.
/// Uses a greedy algorithm: place each mark at the smallest position that
/// maintains the Golomb property (all pairwise distances unique).
use std::collections::HashSet;

use crate::primality::is_prime_bpsw;

/// Construct a valid (non-optimal) Golomb ruler with `n` marks using greedy placement.
/// Returns marks in ascending order. Length is O(n^2) in practice,
/// good enough to seed the search bound.
pub fn construct_golomb(n: usize) -> Vec<u32> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }

    let mut marks = vec![0u32];
    let mut used: HashSet<u32> = HashSet::new();
    let mut pos = 0u32;

    for _ in 1..n {
        let mut gap = 1u32;
        loop {
            let new_pos = pos + gap;
            let mut conflict = false;
            for &m in &marks {
                if used.contains(&(new_pos - m)) {
                    conflict = true;
                    break;
                }
            }
            if !conflict {
                for &m in &marks {
                    used.insert(new_pos - m);
                }
                marks.push(new_pos);
                pos = new_pos;
                break;
            }
            gap += 1;
        }
    }

    marks
}

/// Construct a Golomb ruler using a prime-gap heuristic.
/// Tries prime gaps first (via BPSW primality test) before falling back to
/// composites. This often produces shorter rulers than pure greedy for small n,
/// since prime gaps tend to create fewer distance collisions.
pub fn construct_prime_greedy(n: usize) -> Vec<u32> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }

    let mut marks = vec![0u32];
    let mut used: HashSet<u32> = HashSet::new();
    let mut pos = 0u32;

    for _ in 1..n {
        // Try prime gaps first, then fill in composites
        let mut gap = 2u32; // start from 2 (smallest prime)
        let mut found = false;
        while gap <= pos + 1 {
            if is_prime_bpsw(gap as u64) {
                let new_pos = pos + gap;
                let mut conflict = false;
                for &m in &marks {
                    if used.contains(&(new_pos - m)) {
                        conflict = true;
                        break;
                    }
                }
                if !conflict {
                    for &m in &marks {
                        used.insert(new_pos - m);
                    }
                    marks.push(new_pos);
                    pos = new_pos;
                    found = true;
                    break;
                }
            }
            gap += 1;
        }

        // Fall back to standard greedy if no prime gap worked
        if !found {
            let mut gap = 1u32;
            loop {
                let new_pos = pos + gap;
                let mut conflict = false;
                for &m in &marks {
                    if used.contains(&(new_pos - m)) {
                        conflict = true;
                        break;
                    }
                }
                if !conflict {
                    for &m in &marks {
                        used.insert(new_pos - m);
                    }
                    marks.push(new_pos);
                    pos = new_pos;
                    break;
                }
                gap += 1;
            }
        }
    }

    marks
}

/// Construct a ruler and return its length using the best available method.
/// Tries both greedy and prime-gap constructions and returns the shorter one.
pub fn construct_bound(n: usize) -> u32 {
    let greedy_marks = construct_golomb(n);
    let prime_marks = construct_prime_greedy(n);
    let greedy_len = *greedy_marks.last().unwrap_or(&0);
    let prime_len = *prime_marks.last().unwrap_or(&0);
    greedy_len.min(prime_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construct_valid_golomb() {
        for n in 2..=20 {
            let marks = construct_golomb(n);
            assert_eq!(marks.len(), n);
            assert_eq!(marks[0], 0);
            let mut dists = HashSet::new();
            for i in 0..marks.len() {
                for j in i + 1..marks.len() {
                    let d = marks[j] - marks[i];
                    assert!(dists.insert(d), "Duplicate distance {} for n={}", d, n);
                }
            }
        }
    }

    #[test]
    fn test_construct_prime_greedy_valid() {
        for n in 2..=15 {
            let marks = construct_prime_greedy(n);
            assert_eq!(marks.len(), n);
            assert_eq!(marks[0], 0);
            let mut dists = HashSet::new();
            for i in 0..marks.len() {
                for j in i + 1..marks.len() {
                    let d = marks[j] - marks[i];
                    assert!(dists.insert(d), "Duplicate distance {} for n={}", d, n);
                }
            }
        }
    }

    #[test]
    fn test_construct_bounds() {
        let cases = [(10, 55, 200), (14, 127, 400), (20, 283, 1000)];
        for (n, optimal, max_expected) in cases {
            let len = construct_bound(n);
            assert!(
                len > optimal,
                "n={}: len={} should exceed optimal {}",
                n,
                len,
                optimal
            );
            assert!(
                len < max_expected,
                "n={}: len={} should be < {}",
                n,
                len,
                max_expected
            );
            eprintln!("OGR-{}: bound = {} (optimal = {})", n, len, optimal);
        }
    }
}
