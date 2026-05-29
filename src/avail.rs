/// Incremental available-distance cache for the dynamic lower bound (Lock ② advanced).
///
/// Instead of scanning the entire bitmap for every candidate gap, we precompute
/// the sorted list of smallest available distances once per recursion level,
/// then incrementally remove distances for each candidate gap.
///
/// Complexity: O(MAX_AVAIL * |newbits|) per gap vs O(max_bit) with full scan.
/// For OGR-20: ~20 * 10 ≈ 200 ops vs ~283 ops, but constant factor is much better
/// since we avoid bitmap allocation and word iteration.
///
/// Maximum number of available distances to track.
/// For OGR-28 (28 marks), we need at most 27 distances for the bound.
pub const MAX_AVAIL: usize = 30;

/// Sorted array of the smallest available (unset in dist bitmap) distances,
/// with prefix sums for O(1) k-smallest-sum queries.
#[derive(Clone, Copy)]
pub struct AvailDistances {
    /// Sorted ascending: the smallest available distances.
    dists: [u32; MAX_AVAIL],
    /// prefix[i] = dists[0] + dists[1] + ... + dists[i].
    prefix: [u32; MAX_AVAIL],
    /// Number of valid entries in dists/prefix.
    count: usize,
}

impl AvailDistances {
    /// Build from a bitmap: collect the smallest `max_k` unset distances in [1, max_bit].
    pub fn from_bitmap<const W: usize>(
        bitmap: &crate::bitmap::Bitmap<W>,
        max_k: usize,
        max_bit: usize,
    ) -> Self {
        let mut dists = [0u32; MAX_AVAIL];
        let count = bitmap.collect_smallest_unset(max_k.min(MAX_AVAIL), max_bit, &mut dists);
        let mut prefix = [0u32; MAX_AVAIL];
        let mut sum = 0u32;
        for i in 0..count {
            sum += dists[i];
            prefix[i] = sum;
        }
        Self {
            dists,
            prefix,
            count,
        }
    }

    /// Build from base avail, removing the set bits in `newbits` bitmap.
    /// `newbits_count` is the number of set bits in newbits (avoids recomputation).
    pub fn without_bitmap<const W: usize>(base: &Self, newbits: &crate::bitmap::Bitmap<W>) -> Self {
        let mut result = *base;
        // Remove each distance that appears in newbits
        for bit in newbits.iter_set_bits() {
            if bit == 0 {
                continue;
            }
            result.remove_sorted(bit);
        }
        result
    }

    /// Sum of the k smallest available distances.
    /// Returns None if fewer than k distances are available.
    #[inline(always)]
    pub fn sum_k(&self, k: usize) -> Option<u32> {
        if k == 0 {
            return Some(0);
        }
        if k > self.count {
            return None;
        }
        Some(self.prefix[k - 1])
    }

    /// Number of available distances tracked.
    #[inline(always)]
    #[allow(dead_code)]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Remove a value from the sorted array and recompute prefix.
    fn remove_sorted(&mut self, val: u32) {
        // Binary search for val
        let mut lo = 0usize;
        let mut hi = self.count;
        let mut found = false;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.dists[mid] == val {
                // Shift remaining elements left
                self.dists.copy_within(mid + 1..self.count, mid);
                self.count -= 1;
                found = true;
                break;
            } else if self.dists[mid] < val {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if found {
            // Recompute prefix from the removal point
            let mut sum = if lo > 0 { self.prefix[lo - 1] } else { 0u32 };
            for i in lo..self.count {
                sum += self.dists[i];
                self.prefix[i] = sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::Bitmap;

    #[test]
    fn test_avail_from_bitmap() {
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        bm.set_bit(1);
        bm.set_bit(2);
        bm.set_bit(3);
        // Available: 4, 5, 6, 7, ...
        let avail = AvailDistances::from_bitmap(&bm, 5, 100);
        assert_eq!(avail.count, 5);
        assert_eq!(avail.dists[0], 4);
        assert_eq!(avail.dists[1], 5);
        assert_eq!(avail.sum_k(3).unwrap(), 4 + 5 + 6);
    }

    #[test]
    fn test_avail_without() {
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        bm.set_bit(1);
        bm.set_bit(3);
        // Available: 2, 4, 5, 6, 7, ...
        let base = AvailDistances::from_bitmap(&bm, 5, 100);

        let mut nb: Bitmap<2> = Bitmap::ZERO;
        nb.set_bit(4);
        nb.set_bit(6);
        let filtered = AvailDistances::without_bitmap(&base, &nb);
        // Removed 4 and 6, so: 2, 5, 7, ...
        assert_eq!(filtered.dists[0], 2);
        assert_eq!(filtered.dists[1], 5);
        assert_eq!(filtered.dists[2], 7);
        assert_eq!(filtered.sum_k(2).unwrap(), 2 + 5);
    }

    #[test]
    fn test_avail_matches_scan() {
        // Verify incremental matches full scan
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        bm.set_bit(1);
        bm.set_bit(5);
        bm.set_bit(10);

        let base = AvailDistances::from_bitmap(&bm, 10, 50);

        let mut nb: Bitmap<2> = Bitmap::ZERO;
        nb.set_bit(3);
        nb.set_bit(7);

        let filtered = AvailDistances::without_bitmap(&base, &nb);

        // Full scan version
        let mut test = bm;
        test |= nb;
        let scan_sum = test.sum_smallest_unset(5, 50).unwrap();

        assert_eq!(filtered.sum_k(5).unwrap(), scan_sum);
    }
}
