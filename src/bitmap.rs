/// Multi-word bitmap with optimized shift operations.
/// Core data structure for Phases 2-5.
/// Includes Phase 5 Lock ①: branchless cross-word shift (shl_into)
/// with corrected edge cases for bit_off=0 and g>=64.

use std::ops::{BitAnd, BitOr, BitOrAssign, Not};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bitmap<const W: usize> {
    words: [u64; W],
}

impl<const W: usize> Default for Bitmap<W> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<const W: usize> Bitmap<W> {
    pub const ZERO: Self = Self { words: [0; W] };

    pub const fn one() -> Self {
        let mut words = [0u64; W];
        words[0] = 1;
        Self { words }
    }

    #[inline(always)]
    pub fn set_bit(&mut self, bit: usize) {
        let word = bit / 64;
        let offset = bit % 64;
        if word < W {
            self.words[word] |= 1u64 << offset;
        }
    }

    #[inline(always)]
    pub fn get_bit(&self, bit: usize) -> bool {
        let word = bit / 64;
        let offset = bit % 64;
        if word < W {
            (self.words[word] >> offset) & 1 == 1
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn clear_bit(&mut self, bit: usize) {
        let word = bit / 64;
        let offset = bit % 64;
        if word < W {
            self.words[word] &= !(1u64 << offset);
        }
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        let mut acc = 0u64;
        for i in 0..W {
            acc |= self.words[i];
        }
        acc == 0
    }

    /// Check if any bit is set in both self and other.
    #[inline(always)]
    pub fn intersects(&self, other: &Self) -> bool {
        for i in 0..W {
            if (self.words[i] & other.words[i]) != 0 {
                return true;
            }
        }
        false
    }

    /// Count trailing zeros (index of lowest set bit), or None if zero.
    pub fn ctz(&self) -> Option<u32> {
        for i in 0..W {
            if self.words[i] != 0 {
                return Some(i as u32 * 64 + self.words[i].trailing_zeros());
            }
        }
        None
    }

    /// Iterate over set bit positions.
    pub fn iter_set_bits(&self) -> BitmapIter<'_, W> {
        BitmapIter::new(self)
    }

    /// Count number of set bits.
    pub fn count_ones(&self) -> u32 {
        let mut count = 0u32;
        for i in 0..W {
            count += self.words[i].count_ones();
        }
        count
    }

    /// Shift left by `g` bits into a new bitmap.
    #[inline(always)]
    pub fn shl(&self, g: usize) -> Self {
        let mut result = Self::ZERO;
        self.shl_into(g, &mut result);
        result
    }

    /// Shift left by `g` bits, writing result into `dst`.
    ///
    /// Handles three cases:
    /// 1. g >= W*64: result is zero
    /// 2. bit_off == 0: pure word-aligned copy
    /// 3. bit_off != 0: cross-word shift with carry
    ///
    /// Phase 5 Lock ①: avoids `>>(64-g)` overflow when bit_off=0
    /// by handling that case separately. Also handles g>=64 via
    /// word_off decomposition.
    #[inline(always)]
    pub fn shl_into(&self, g: usize, dst: &mut Self) {
        if g == 0 {
            *dst = *self;
            return;
        }

        let word_off = g / 64;
        let bit_off = g % 64;

        if word_off >= W {
            *dst = Self::ZERO;
            return;
        }

        if bit_off == 0 {
            // Pure word-aligned shift — no bit-level carry needed.
            // This avoids the `>>(64-0)` overflow trap.
            *dst = Self::ZERO;
            let mut i = W;
            while i > word_off {
                i -= 1;
                dst.words[i] = self.words[i - word_off];
            }
        } else {
            // Cross-word shift with carry.
            // bit_off is in 1..63, so inv = 64 - bit_off is in 1..63 — safe for shift.
            *dst = Self::ZERO;
            let inv = 64 - bit_off;
            let mut i = word_off;
            while i < W {
                let src = i - word_off;
                dst.words[i] = self.words[src] << bit_off;
                if src > 0 {
                    dst.words[i] |= self.words[src - 1] >> inv;
                }
                i += 1;
            }
        }
    }

    /// Find the sum of the `k` smallest cleared bit positions in the bitmap.
    /// Uses word-level trailing_zeros on complement for fast scanning.
    /// Returns None if fewer than k cleared bits exist in [1, max_bit].
    #[inline]
    pub fn sum_smallest_unset(&self, k: usize, max_bit: usize) -> Option<u32> {
        if k == 0 {
            return Some(0);
        }
        let mut sum: u32 = 0;
        let mut count = 0;
        for word_idx in 0..W {
            let base = word_idx * 64;
            if base > max_bit {
                break;
            }
            // Complement: unset bits become 1
            let mut unset = !self.words[word_idx];
            // Skip distance 0 (bit 0 of word 0)
            if word_idx == 0 {
                unset &= !1u64;
            }
            // Mask out bits beyond max_bit
            let word_end = ((word_idx + 1) * 64).min(max_bit + 1);
            let bits_in_word = word_end - base;
            if bits_in_word < 64 {
                unset &= (1u64 << bits_in_word) - 1;
            }
            // Fast iteration using trailing_zeros
            while unset != 0 {
                let bit = unset.trailing_zeros() as usize;
                sum += (base + bit) as u32;
                count += 1;
                if count == k {
                    return Some(sum);
                }
                unset &= unset - 1; // clear lowest set bit
            }
        }
        if count >= k {
            Some(sum)
        } else {
            None
        }
    }

    /// Collect the `k` smallest cleared bit positions into a sorted array.
    /// Returns the number of positions actually collected (may be < k).
    #[inline]
    pub fn collect_smallest_unset(&self, k: usize, max_bit: usize, out: &mut [u32]) -> usize {
        let mut count = 0;
        for word_idx in 0..W {
            let base = word_idx * 64;
            if base > max_bit || count >= k {
                break;
            }
            let mut unset = !self.words[word_idx];
            if word_idx == 0 {
                unset &= !1u64;
            }
            let word_end = ((word_idx + 1) * 64).min(max_bit + 1);
            let bits_in_word = word_end - base;
            if bits_in_word < 64 {
                unset &= (1u64 << bits_in_word) - 1;
            }
            while unset != 0 {
                let bit = unset.trailing_zeros() as usize;
                out[count] = (base + bit) as u32;
                count += 1;
                if count >= k {
                    return count;
                }
                unset &= unset - 1;
            }
        }
        count
    }

    #[inline(always)]
    pub fn word(&self, i: usize) -> u64 {
        self.words[i]
    }
}

pub struct BitmapIter<'a, const W: usize> {
    bitmap: &'a Bitmap<W>,
    word_idx: usize,
    current: u64,
}

impl<'a, const W: usize> BitmapIter<'a, W> {
    fn new(bitmap: &'a Bitmap<W>) -> Self {
        Self {
            bitmap,
            word_idx: 0,
            current: if W > 0 { bitmap.words[0] } else { 0 },
        }
    }
}

impl<'a, const W: usize> Iterator for BitmapIter<'a, W> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current != 0 {
                let bit = self.current.trailing_zeros();
                self.current &= self.current - 1;
                return Some(self.word_idx as u32 * 64 + bit);
            }
            self.word_idx += 1;
            if self.word_idx >= W {
                return None;
            }
            self.current = self.bitmap.words[self.word_idx];
        }
    }
}

impl<const W: usize> BitOr for Bitmap<W> {
    type Output = Self;
    #[inline(always)]
    fn bitor(mut self, rhs: Self) -> Self {
        self |= rhs;
        self
    }
}

impl<const W: usize> BitOrAssign for Bitmap<W> {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) {
        for i in 0..W {
            self.words[i] |= rhs.words[i];
        }
    }
}

impl<const W: usize> BitAnd for Bitmap<W> {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..W {
            result.words[i] = self.words[i] & rhs.words[i];
        }
        result
    }
}

impl<const W: usize> Not for Bitmap<W> {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        let mut result = Self::ZERO;
        for i in 0..W {
            result.words[i] = !self.words[i];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_get_bit() {
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        assert!(!bm.get_bit(0));
        assert!(!bm.get_bit(100));

        bm.set_bit(0);
        bm.set_bit(63);
        bm.set_bit(64);
        bm.set_bit(127);

        assert!(bm.get_bit(0));
        assert!(bm.get_bit(63));
        assert!(bm.get_bit(64));
        assert!(bm.get_bit(127));
        assert!(!bm.get_bit(1));
        assert!(!bm.get_bit(62));
    }

    #[test]
    fn test_shl_zero() {
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        bm.set_bit(0);
        bm.set_bit(5);
        bm.set_bit(63);

        let shifted = bm.shl(0);
        assert_eq!(shifted, bm);
    }

    #[test]
    fn test_shl_small() {
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        bm.set_bit(0);
        // 1 << 3 = bit 3
        let shifted = bm.shl(3);
        assert!(shifted.get_bit(3));
        assert!(!shifted.get_bit(0));
    }

    #[test]
    fn test_shl_cross_word() {
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        bm.set_bit(62); // bit 62 in word 0
        // shift by 3: bit 62 -> bit 65 (word 1, offset 1)
        let shifted = bm.shl(3);
        assert!(!shifted.get_bit(62));
        assert!(shifted.get_bit(65));
    }

    #[test]
    fn test_shl_word_aligned() {
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        bm.set_bit(0);
        bm.set_bit(5);
        // shift by 64 (exactly one word)
        let shifted = bm.shl(64);
        assert!(!shifted.get_bit(0));
        assert!(!shifted.get_bit(5));
        assert!(shifted.get_bit(64));
        assert!(shifted.get_bit(69));
    }

    #[test]
    fn test_shl_beyond_size() {
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        bm.set_bit(0);
        let shifted = bm.shl(128);
        assert!(shifted.is_zero());
    }

    #[test]
    fn test_shl_large_gap() {
        // Test g >= 64 (Phase 5 Lock ① edge case)
        let mut bm: Bitmap<5> = Bitmap::ZERO;
        bm.set_bit(0);
        bm.set_bit(10);
        bm.set_bit(50);

        // Shift by 130 = 2*64 + 2
        let shifted = bm.shl(130);
        assert!(shifted.get_bit(130)); // 0 + 130
        assert!(shifted.get_bit(140)); // 10 + 130
        assert!(shifted.get_bit(180)); // 50 + 130
        assert!(!shifted.get_bit(0));
        assert!(!shifted.get_bit(10));
    }

    #[test]
    fn test_intersects() {
        let mut a: Bitmap<2> = Bitmap::ZERO;
        let mut b: Bitmap<2> = Bitmap::ZERO;
        assert!(!a.intersects(&b));

        a.set_bit(5);
        b.set_bit(10);
        assert!(!a.intersects(&b));

        b.set_bit(5);
        assert!(a.intersects(&b));
    }

    #[test]
    fn test_iter_set_bits() {
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        bm.set_bit(0);
        bm.set_bit(3);
        bm.set_bit(64);
        bm.set_bit(100);

        let bits: Vec<u32> = bm.iter_set_bits().collect();
        assert_eq!(bits, vec![0, 3, 64, 100]);
    }

    #[test]
    fn test_sum_smallest_unset() {
        let mut bm: Bitmap<2> = Bitmap::ZERO;
        bm.set_bit(1);
        bm.set_bit(2);
        bm.set_bit(3);
        bm.set_bit(4);
        // Smallest unset bits: 5, 6, 7, ...
        let sum = bm.sum_smallest_unset(3, 100).unwrap();
        assert_eq!(sum, 5 + 6 + 7);
    }

    #[test]
    fn test_shl_into_vs_shl() {
        // Exhaustive check: shl_into must match shl for various g values
        let mut bm: Bitmap<5> = Bitmap::ZERO;
        bm.set_bit(0);
        bm.set_bit(7);
        bm.set_bit(42);
        bm.set_bit(63);
        bm.set_bit(100);
        bm.set_bit(200);

        for g in [0, 1, 2, 7, 63, 64, 65, 127, 128, 130, 200, 319] {
            let expected = bm.shl(g);
            let mut actual = Bitmap::<5>::ZERO;
            bm.shl_into(g, &mut actual);
            assert_eq!(actual, expected, "Mismatch at g={}", g);
        }
    }
}
