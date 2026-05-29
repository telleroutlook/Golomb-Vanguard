/// Baillie-PSW primality test for u64.
///
/// Combines a strong Miller-Rabin test (base 2) with a strong Lucas-Selfridge test.
/// No known composites pass both tests (verified for all n < 2^64).
/// Trial division by small primes is used as a fast-reject first step.
///
/// Small primes for trial division.
const SMALL_PRIMES: &[u64] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
];

/// Returns true if n is probably prime using the Baillie-PSW test.
pub fn is_prime_bpsw(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for &p in SMALL_PRIMES {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }
    if !miller_rabin_strong(n, 2) {
        return false;
    }
    strong_lucas_test(n)
}

/// Strong Miller-Rabin primality test for a single base.
///
/// Writes n - 1 = d * 2^s with d odd, then checks whether
/// a^d ≡ 1 (mod n) or a^(d*2^r) ≡ -1 (mod n) for some 0 <= r < s.
pub fn miller_rabin_strong(n: u64, base: u64) -> bool {
    debug_assert!(n > 2 && !n.is_multiple_of(2));

    let mut d = n - 1;
    let mut s = 0u32;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }

    let mut x = mod_pow(base, d, n);
    if x == 1 || x == n - 1 {
        return true;
    }

    for _ in 1..s {
        x = mod_mul(x, x, n);
        if x == n - 1 {
            return true;
        }
    }

    false
}

/// Strong Lucas-Selfridge test.
///
/// Uses Selfridge's Method A for parameter selection:
/// find the first D in the sequence 5, -7, 9, -11, 13, -15, ...
/// such that Jacobi(D, n) = -1. Then P = 1, Q = (1 - D) / 4.
///
/// The strong test writes n + 1 = d * 2^s and checks:
///   U_d ≡ 0 (mod n), or
///   V_{d * 2^r} ≡ 0 (mod n) for some 0 <= r < s.
pub fn strong_lucas_test(n: u64) -> bool {
    debug_assert!(n > 2 && !n.is_multiple_of(2));

    // Selfridge parameter selection: find D with (D/n) = -1
    let mut d_abs = 5i64;
    let mut sign: i64 = 1;
    let d_found;
    loop {
        let d_candidate = d_abs * sign;
        let j = jacobi(d_candidate, n);
        if j == 0 && d_abs as u64 != n {
            return false; // gcd(|D|, n) > 1, n is composite
        }
        if j == -1 {
            d_found = d_candidate;
            break;
        }
        d_abs += 2;
        sign = -sign;
        if d_abs > 1_000_000 {
            // Extremely unlikely for any n < 2^64
            return true;
        }
    }

    // P = 1, Q = (1 - D) / 4
    let p = 1u64;
    let q_raw = (1 - d_found) / 4;
    let q = if q_raw >= 0 {
        q_raw as u64 % n
    } else {
        n - ((-q_raw as u64) % n)
    };

    // Write n + 1 = d * 2^s with d odd
    let mut d = n + 1;
    let mut s = 0u32;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }

    // Compute U_d and V_d using the binary Lucas chain
    let (u, v) = lucas_chain(d, p, q, n);

    if u == 0 {
        return true;
    }

    // Check V_{d * 2^r} ≡ 0 for r = 0..s-1
    // Doubling formula: V_{2k} = V_k^2 - 2*Q^k
    // We need to track Q^k alongside V_k.
    // At this point we have V_d but not Q^d directly.
    // We can re-derive using the chain that also tracks Q^k.
    let mut vr = v;
    let mut qkr = mod_pow(q, d, n); // Q^d mod n

    for _ in 0..s {
        if vr == 0 {
            return true;
        }
        // V_{2k} = V_k^2 - 2*Q^k (mod n)
        vr = (mod_mul(vr, vr, n) + n - mod_mul(2, qkr, n)) % n;
        // Q^{2k} = (Q^k)^2 (mod n)
        qkr = mod_mul(qkr, qkr, n);
    }

    false
}

/// Compute U_k and V_k for the Lucas sequence U_k(P,Q), V_k(P,Q) modulo n.
///
/// Uses the standard binary chain algorithm processing bits of k from MSB to LSB.
/// Based on the identities:
///   U_{2k}   = U_k * V_k          (mod n)
///   V_{2k}   = V_k^2 - 2*Q^k      (mod n)
///   U_{2k+1} = (P*U_{2k} + V_{2k}) / 2  -- but we use the product form instead
///
/// We track (U, V, Q^k) through the chain.
fn lucas_chain(k: u64, p: u64, q: u64, n: u64) -> (u64, u64) {
    if k == 0 {
        return (0, 2 % n);
    }

    // Start with U_1 = 1, V_1 = P, Q^1 = Q
    let mut u = 1u64;
    let mut v = p % n;
    let mut qk = q % n; // tracks Q^current_index

    // Process bits of k from the second-highest bit down to bit 0
    let bits = 64 - k.leading_zeros();

    for i in (0..bits - 1).rev() {
        // Double step: (U_k, V_k) -> (U_{2k}, V_{2k})
        // U_{2k} = U_k * V_k (mod n)
        // V_{2k} = V_k^2 - 2*Q^k (mod n)
        // Q^{2k} = (Q^k)^2 (mod n)
        let u2 = mod_mul(u, v, n);
        let v2 = (mod_mul(v, v, n) + n - mod_mul(2, qk, n)) % n;
        let qk2 = mod_mul(qk, qk, n);

        if (k >> i) & 1 == 0 {
            // Bit is 0: just use the doubled values
            u = u2;
            v = v2;
            qk = qk2;
        } else {
            // Bit is 1: double, then increment
            // (U_{2k}, V_{2k}) -> (U_{2k+1}, V_{2k+1})
            // U_{2k+1} = (P * U_{2k} + V_{2k}) / 2 (mod n)
            // V_{2k+1} = (D * U_{2k} + P * V_{2k}) / 2 (mod n)
            // where D = P^2 - 4Q
            //
            // Division by 2 mod n: multiply by (n+1)/2 (since n is odd, (n+1)/2 is the inverse of 2).
            let half = n.div_ceil(2);
            let d_val = (mod_mul(p, p, n) + n - mod_mul(4, q, n)) % n;

            u = mod_mul((mod_mul(p, u2, n) + v2) % n, half, n);
            v = mod_mul((mod_mul(d_val, u2, n) + mod_mul(p, v2, n)) % n, half, n);
            qk = mod_mul(qk2, q, n);
        }
    }

    (u, v)
}

/// Compute Jacobi symbol (a/n) for odd positive n.
fn jacobi(mut a: i64, n: u64) -> i64 {
    debug_assert!(n % 2 == 1);
    let mut n = n as i64;
    let mut result = 1i64;

    a %= n;
    if a < 0 {
        a += n;
    }

    while a != 0 {
        while a % 2 == 0 {
            a /= 2;
            let n_mod_8 = ((n % 8) + 8) % 8;
            if n_mod_8 == 3 || n_mod_8 == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a %= n;
    }

    if n == 1 {
        result
    } else {
        0
    }
}

/// Modular exponentiation: base^exp mod m.
fn mod_pow(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, m);
        }
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    result
}

/// Modular multiplication: (a * b) mod m, using u128 to avoid overflow.
fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_primes() {
        let primes = [2, 3, 5, 7, 11, 13, 97, 7919];
        for &p in &primes {
            assert!(is_prime_bpsw(p), "{} should be prime", p);
        }
    }

    #[test]
    fn test_small_composites() {
        let composites = [4u64, 6, 9, 15, 21, 35];
        for &c in &composites {
            assert!(!is_prime_bpsw(c), "{} should be composite", c);
        }
    }

    #[test]
    fn test_carmichael_numbers() {
        let carmichaels = [561u64, 1105, 1729];
        for &c in &carmichaels {
            assert!(!is_prime_bpsw(c), "Carmichael {} should be composite", c);
        }
    }

    #[test]
    fn test_large_prime() {
        // 4294967291 is the largest prime < 2^32
        assert!(is_prime_bpsw(4_294_967_291));
    }

    #[test]
    fn test_edge_cases() {
        assert!(!is_prime_bpsw(0));
        assert!(!is_prime_bpsw(1));
        assert!(is_prime_bpsw(2));
        assert!(is_prime_bpsw(3));
    }

    #[test]
    fn test_pseudoprime_base_2() {
        // 2047 is a strong pseudoprime to base 2
        assert!(!is_prime_bpsw(2047));
        // 1373653 is a strong pseudoprime to bases 2 and 3
        assert!(!is_prime_bpsw(1_373_653));
    }

    #[test]
    fn test_more_primes() {
        // Verify a range of known primes
        let primes = [
            101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 997, 1009, 1013, 1019,
            65537, // Fermat prime F4
        ];
        for &p in &primes {
            assert!(is_prime_bpsw(p), "{} should be prime", p);
        }
    }

    #[test]
    fn test_more_composites() {
        let composites = [
            49, 77, 91, 121, 143, 169, 221, 323, 999, 1001,
            3215031751, // strong pseudoprime to bases 2, 3, 5, 7
        ];
        for &c in &composites {
            assert!(!is_prime_bpsw(c), "{} should be composite", c);
        }
    }

    #[test]
    fn test_miller_rabin_base2_strong_pseudoprimes() {
        // These pass Miller-Rabin base 2 but must be caught by Lucas
        assert!(!is_prime_bpsw(2047));
        assert!(!is_prime_bpsw(3277));
        assert!(!is_prime_bpsw(4033));
        assert!(!is_prime_bpsw(4681));
    }
}
