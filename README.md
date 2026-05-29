<div align="center">

# Golomb-Vanguard

**High-Performance Optimal Golomb Ruler Search Engine**

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Find Optimal Golomb Rulers (OGRs) with progressively optimized search engines — from naive DFS to parallel work-stealing.

> **Scope:** This is a single-machine educational engine. It can find OGR-n for n ≤ ~13 in seconds and verify known OGRs up to n=28. It is **not** a contender for n ≥ 29, where OGR search requires distributed computing (the OGR-28 record was set by distributed.net over millions of CPU-years).

</div>

---

## What is a Golomb Ruler?

A Golomb ruler with `n` marks has all pairwise distances unique. An **Optimal** Golomb Ruler (OGR) has the shortest possible length for `n` marks.

```
OGR-4: 0  1  4  6    (length 6)
       ├──┤     │
       1  3  2     ← all distances unique: {1,2,3,4,5,6}
          ├──────┤
          4  2
```

Optimal lengths for `n = 2..28` are known (OEIS A003022), mostly found by the [distributed.net OGR-NG project](https://www.distributed.net/OGR). For `n >= 29`, they remain **open research problems** requiring massive distributed search.

---

## Engine Versions

| Engine | Algorithm | Strength |
|:-------|:----------|:---------|
| **v1** | Naive DFS | Baseline reference implementation |
| **v2** | Bitmask | Bit manipulation for fast distance checking |
| **v3** | Branch & Bound | Pruning + symmetry breaking |
| **v4** | Parallel Ultimate | Multi-threaded work-stealing with Rayon |

Each version builds on the previous, adding sophistication without sacrificing correctness.

---

## Quick Start

```bash
git clone https://github.com/telleroutlook/Golomb-Vanguard.git
cd Golomb-Vanguard
cargo build --release
```

```bash
# Find OGR-8 using the best engine (v4)
./target/release/golomb_vanguard -n 8

# Find OGR-12 with 4 threads
./target/release/golomb_vanguard -n 12 -t 4

# Prove optimality for OGR-15
./target/release/golomb_vanguard -n 15 -m prove

# Use a specific engine
./target/release/golomb_vanguard -n 10 -e v3
```

---

## Known Optimal Lengths

| Marks | Optimal Length | Marks | Optimal Length |
|:------|:--------------|:------|:--------------|
| 2 | 1 | 16 | 177 |
| 3 | 3 | 17 | 199 |
| 4 | 6 | 18 | 216 |
| 5 | 11 | 19 | 246 |
| 6 | 17 | 20 | 283 |
| 7 | 25 | 21 | 333 |
| 8 | 34 | 22 | 356 |
| 9 | 44 | 23 | 372 |
| 10 | 55 | 24 | 425 |
| 11 | 72 | 25 | 480 |
| 12 | 85 | 26 | 492 |
| 13 | 106 | 27 | 553 |
| 14 | 127 | 28 | 585 |
| 15 | 151 | **29+** | **Open** |

---

## Technical Highlights

- **Multi-word bitmap** — cache-line aligned, branchless distance tracking
- **Symmetry breaking** — exploits ruler mirror symmetry to halve the search space
- **Greedy bounds** — constructive upper bounds for pruning
- **Parallel work-stealing** — Rayon-based, scales across all cores
- **CPU-specific optimizations** — `target-cpu=native`, LTO, single codegen unit

---

## Repository Structure

```
Golomb-Vanguard/
├── src/
│   ├── main.rs          # CLI entry point
│   ├── known.rs         # Known optimal lengths (OEIS A003022)
│   ├── naive.rs         # v1: Naive DFS
│   ├── bitmap.rs        # Multi-word bitmap data structure
│   ├── avail.rs         # Distance tracking utilities
│   ├── construct.rs     # Greedy construction for initial bounds
│   ├── engine_v2.rs     # v2: Bitmask engine
│   ├── engine_v3.rs     # v3: Branch & bound
│   └── engine_v4.rs     # v4: Parallel ultimate
├── .cargo/config.toml   # Build optimizations
└── Cargo.toml
```

---

## Applications

Gardener's problem is more than a mathematical curiosity — OGRs appear in:

- **Coding theory** — optimal error-correcting codes
- **Frequency allocation** — interference-free channel spacing
- **Radar and sonar** — pulse sequence design
- **Radio astronomy** — aperture synthesis optimization

---

## License

Apache License 2.0
