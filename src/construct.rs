/// Constructive Golomb ruler generation for seeding find-mode upper bound.
/// Uses a greedy algorithm: place each mark at the smallest position that
/// maintains the Golomb property (all pairwise distances unique).

use std::collections::HashSet;

/// Construct a valid (non-optimal) Golomb ruler with `n` marks.
/// Returns marks in ascending order. Length is O(n²) in practice,
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

/// Construct a ruler and return its length.
pub fn construct_bound(n: usize) -> u32 {
    let marks = construct_golomb(n);
    *marks.last().unwrap_or(&0)
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
    fn test_construct_bounds() {
        // Greedy construction should produce lengths much better than n³
        let cases = [
            (10, 55, 200),  // known optimal = 55, expect < 200
            (14, 127, 400),
            (20, 283, 1000),
        ];
        for (n, optimal, max_expected) in cases {
            let len = construct_bound(n);
            assert!(len > optimal, "n={}: greedy len={} should exceed optimal {}", n, len, optimal);
            assert!(len < max_expected, "n={}: greedy len={} should be < {}", n, len, max_expected);
            eprintln!("OGR-{}: greedy bound = {} (optimal = {})", n, len, optimal);
        }
    }
}
