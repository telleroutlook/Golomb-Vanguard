mod known;
mod naive;
mod bitmap;
mod avail;
mod construct;
mod primality;
mod engine_v2;
mod engine_v3;
mod engine_v4;

use clap::Parser;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "golomb-vanguard", about = "Optimal Golomb Ruler ultra-fast search engine")]
struct Cli {
    /// Number of marks
    #[arg(short, long)]
    n: usize,

    /// Engine version: v1 (naive), v2 (bitmask), v3 (branch&bound), v4 (parallel)
    #[arg(short, long, default_value = "v4")]
    engine: String,

    /// Mode: find (search for ruler) or prove (prove optimality)
    #[arg(short, long, default_value = "find")]
    mode: String,

    /// Maximum ruler length (default: known optimal)
    #[arg(long)]
    max_len: Option<u32>,

    /// Number of threads (v4 only, default: all cores)
    #[arg(short, long)]
    threads: Option<usize>,
}

fn main() {
    let cli = Cli::parse();

    let n = cli.n;
    if n < 2 {
        eprintln!("n must be >= 2");
        std::process::exit(1);
    }

    let known = known::optimal_length(n);
    let max_len = cli.max_len.or(known).unwrap_or_else(|| {
        let bound = construct::construct_bound(n);
        eprintln!("No known optimal for OGR-{}, using constructive bound: {}", n, bound);
        bound
    });

    let threads = cli.threads.unwrap_or_else(|| {
        std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4)
    });

    eprintln!("OGR-{} | engine={} | mode={} | max_len={} | threads={}",
              n, cli.engine, cli.mode, max_len, threads);

    match cli.mode.as_str() {
        "find" => run_find(n, max_len, &cli.engine, threads),
        "prove" => run_prove(n, max_len, &cli.engine, threads),
        _ => {
            eprintln!("Unknown mode '{}'. Use 'find' or 'prove'.", cli.mode);
            std::process::exit(1);
        }
    }
}

fn run_find(n: usize, max_len: u32, engine: &str, threads: usize) {
    let start = Instant::now();

    let result = match engine {
        "v1" => {
            eprintln!("Phase 1: Naive DFS");
            naive::find(n, max_len).map(|m| (m.last().copied().unwrap_or(0), m))
        }
        "v2" => {
            eprintln!("Phase 2: Bitmask engine");
            engine_v2::find_dispatched(n, max_len).map(|m| (*m.last().unwrap(), m))
        }
        "v3" => {
            eprintln!("Phase 3: Branch & bound");
            engine_v3::find_optimal_dispatched(n, max_len)
        }
        "v4" => {
            eprintln!("Phase 4: Parallel ultimate");
            engine_v4::find_optimal_dispatched(n, max_len, threads)
        }
        _ => {
            eprintln!("Unknown engine '{}'. Use v1, v2, v3, or v4.", engine);
            std::process::exit(1);
        }
    };

    let elapsed = start.elapsed();

    match result {
        Some((len, marks)) => {
            println!("FOUND: OGR-{} length = {}", n, len);
            println!("Marks: {:?}", marks);
            if let Some(expected) = known::optimal_length(n) {
                if len == expected {
                    println!("MATCHES known optimal value ({})", expected);
                } else {
                    println!("WARNING: known optimal is {}, found {}", expected, len);
                }
            }
            eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
        }
        None => {
            println!("No Golomb ruler with {} marks of length <= {} exists.", n, max_len);
            eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
        }
    }
}

fn run_prove(n: usize, max_len: u32, engine: &str, threads: usize) {
    // find_optimal exhaustively searches all rulers ≤ max_len.
    // If the shortest found equals max_len, optimality is proven
    // (no shorter ruler exists — the search was exhaustive).
    eprintln!("Proving OGR-{} optimal: exhaustive search at length {}...", n, max_len);
    let start = Instant::now();

    let found = match engine {
        "v1" => naive::find(n, max_len).map(|m| (*m.last().unwrap(), m)),
        "v2" => engine_v2::find_dispatched(n, max_len).map(|m| (*m.last().unwrap(), m)),
        "v3" => engine_v3::find_optimal_dispatched(n, max_len),
        "v4" => engine_v4::find_optimal_dispatched(n, max_len, threads),
        _ => {
            eprintln!("Unknown engine '{}'.", engine);
            std::process::exit(1);
        }
    };

    let elapsed = start.elapsed();

    match found {
        Some((len, marks)) => {
            if len == max_len {
                println!("PROVEN: OGR-{} = {} (optimal, exhaustive search found no shorter ruler)", n, max_len);
                println!("Marks: {:?}", marks);
            } else {
                println!("NOT PROVEN: found shorter ruler (length {} < {})", len, max_len);
                println!("Marks: {:?}", marks);
            }
        }
        None => {
            println!("No ruler with {} marks of length <= {} exists.", n, max_len);
        }
    }
    eprintln!("Time: {:.3}s", elapsed.as_secs_f64());
}
