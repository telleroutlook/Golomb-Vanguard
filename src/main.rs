mod known;
mod naive;
mod bitmap;
mod avail;
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
        eprintln!("No known optimal value for OGR-{} and no --max-len specified", n);
        std::process::exit(1);
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
    // First, verify we can find the known optimal
    eprintln!("Step 1: Verifying ruler exists at length {}...", max_len);
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

    match found {
        Some((len, marks)) => {
            println!("Found ruler at length {}: {:?}", len, marks);
        }
        None => {
            println!("No ruler found at length {} — already proven impossible!", max_len);
            return;
        }
    }
    eprintln!("Find time: {:.3}s", start.elapsed().as_secs_f64());

    // Now prove length-1 is impossible
    if max_len == 0 {
        eprintln!("Cannot prove below length 0.");
        return;
    }
    let prove_len = max_len - 1;
    eprintln!("Step 2: Proving no ruler exists at length {}...", prove_len);

    let prove_start = Instant::now();
    let impossible = match engine {
        "v1" => !naive::exists(n, prove_len),
        "v2" => !engine_v2::exists_dispatched(n, prove_len),
        "v3" => engine_v3::prove_impossible_dispatched(n, prove_len),
        "v4" => engine_v4::prove_impossible_dispatched(n, prove_len, threads),
        _ => false,
    };
    let prove_elapsed = prove_start.elapsed();

    if impossible {
        println!("PROVEN: OGR-{} = {} (optimal, no shorter ruler exists)", n, max_len);
    } else {
        println!("NOT PROVEN: A ruler with {} marks of length <= {} may exist.", n, prove_len);
    }
    eprintln!("Proof time: {:.3}s", prove_elapsed.as_secs_f64());
    eprintln!("Total time: {:.3}s", start.elapsed().as_secs_f64());
}
