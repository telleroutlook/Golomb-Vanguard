/// Phase 1: Naive DFS backtracking engine.
/// Pure correctness, no performance tricks. Serves as the golden reference.

/// Find a Golomb ruler with `n` marks of length <= `max_len`.
/// Returns mark positions if found.
pub fn find(n: usize, max_len: u32) -> Option<Vec<u32>> {
    if n == 0 {
        return Some(vec![]);
    }
    if n == 1 {
        return Some(vec![0]);
    }
    let mut marks = vec![0u32];
    let mut used = vec![false; max_len as usize + 1];
    dfs_find(n, max_len, &mut marks, &mut used)
}

/// Exhaustively check whether a Golomb ruler with `n` marks of length <= `max_len` exists.
/// Returns true if it does (NOT proven impossible).
pub fn exists(n: usize, max_len: u32) -> bool {
    if n <= 1 || max_len == 0 {
        return n <= 1;
    }
    let mut marks = vec![0u32];
    let mut used = vec![false; max_len as usize + 1];
    dfs_exists(n, max_len, &mut marks, &mut used)
}

fn dfs_find(
    n: usize,
    max_len: u32,
    marks: &mut Vec<u32>,
    used: &mut Vec<bool>,
) -> Option<Vec<u32>> {
    if marks.len() == n {
        return Some(marks.clone());
    }
    let last = *marks.last().unwrap();
    for x in (last + 1)..=max_len {
        if try_place(x, marks, used) {
            let result = dfs_find(n, max_len, marks, used);
            if result.is_some() {
                return result;
            }
            unplace(x, marks, used);
        }
    }
    None
}

fn dfs_exists(
    n: usize,
    max_len: u32,
    marks: &mut Vec<u32>,
    used: &mut Vec<bool>,
) -> bool {
    if marks.len() == n {
        return true;
    }
    let last = *marks.last().unwrap();
    for x in (last + 1)..=max_len {
        if try_place(x, marks, used) {
            if dfs_exists(n, max_len, marks, used) {
                unplace(x, marks, used);
                return true;
            }
            unplace(x, marks, used);
        }
    }
    false
}

/// Try to place mark at position `x`. Returns true if all new distances are unique.
fn try_place(x: u32, marks: &mut Vec<u32>, used: &mut Vec<bool>) -> bool {
    let mut dists = [0usize; 32];
    let mut count = 0;
    for &m in marks.iter() {
        let d = (x - m) as usize;
        if d >= used.len() || used[d] {
            return false;
        }
        dists[count] = d;
        count += 1;
    }
    for i in 0..count {
        used[dists[i]] = true;
    }
    marks.push(x);
    true
}

/// Remove the last mark and unmark its distances.
fn unplace(x: u32, marks: &mut Vec<u32>, used: &mut Vec<bool>) {
    marks.pop();
    for &m in marks.iter() {
        let d = (x - m) as usize;
        used[d] = false;
    }
}

/// Verify that a set of marks forms a valid Golomb ruler (all pairwise distances unique).
pub fn verify_golomb(marks: &[u32]) {
    let n = marks.len();
    let mut seen = std::collections::HashSet::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = marks[j] - marks[i];
            assert!(seen.insert(d), "duplicate distance {} in {:?}", d, marks);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::known::optimal_length;

    #[test]
    fn test_ogr_small() {
        for n in 2..=9 {
            let expected = optimal_length(n).unwrap();
            let result = find(n, expected).unwrap();
            assert_eq!(*result.last().unwrap(), expected,
                "OGR-{} should have length {}", n, expected);
            verify_golomb(&result);
        }
    }

    #[test]
    fn test_ogr_10() {
        let result = find(10, 55).unwrap();
        assert_eq!(*result.last().unwrap(), 55, "OGR-10 must be 55, not 36!");
        verify_golomb(&result);
    }

    #[test]
    fn test_prove_optimality_small() {
        // Prove OGR-6 (17) is optimal: no ruler of length 16
        assert!(find(6, 16).is_none());
        // But length 17 works
        assert!(find(6, 17).is_some());
    }

    fn verify_golomb(marks: &[u32]) {
        super::verify_golomb(marks);
    }
}
