#[macro_use]
extern crate criterion;

use rand::prelude::*;
use criterion::{Bencher, Criterion, Throughput, BenchmarkId, black_box};

use simulate_seqs::*;
use block_aligner::scan_block::*;
use block_aligner::scores::*;

use bio::alignment::distance::simd::{bounded_levenshtein, levenshtein};

use parasailors::{Matrix, *};

//use libwfa::{affine_wavefront::*, bindings::*, penalties::*};

use rust_wfa2::aligner::*;

use ksw2_sys::*;

use Scrooge_sys::*;

fn lowdivalign_bench(crit: &mut Criterion) {

    for divergence in vec![0.01, 0.025, 0.05, 0.075, 0.1] { 

        let divname = |s| format!("d{}_{}",divergence,s);

        let mut rng = StdRng:: from_entropy();

        // simulate q and r
        let len = 1000;
        let nb_err = ((divergence as f64) * (len as f64)) as usize;
        let r = black_box(rand_str(len, &NUC, &mut rng));
        let q = black_box(rand_mutate(&r, nb_err, &NUC, &mut rng));

        // preparation for block_aligner
        let block_size = 16;
        let run_gaps = Gaps { open: -2, extend: -1 };
        let r_padded = PaddedBytes::from_bytes::<NucMatrix>(&r, 2048);
        let q_padded = PaddedBytes::from_bytes::<NucMatrix>(&q, 2048);

        // preparation for parasailors
        let matrix = Matrix::new(MatrixType::IdentityWithPenalty);
        let profile = parasailors::Profile::new(&q, &matrix);

        // preparation for libwfa
        /*let libwfa_alloc = libwfa::mm_allocator::MMAllocator::new(BUFFER_SIZE_512M as u64);
          let mut penalties = AffinePenalties {
          match_: 0,
          mismatch: 4,
          gap_opening: 6,
          gap_extension: 2,
          };*/

        // preparation for wfa2
        let mut wfa2_aligner = WFAlignerGapAffine::new(4, 6, 2, AlignmentScope::Alignment, MemoryModel::MemoryHigh);
        wfa2_aligner.set_heuristic(Heuristic::BandedAdaptive(-10, 10, 1));

        // preparation for ksw2
        let a = 1;
        let b = -2;
        let mat = [ a,b,b,b,0, b,a,b,b,0, b,b,a,b,0, b,b,b,a,0, 0,0,0,0,0 ];
        let mut ez : ksw_extz_t = unsafe { std::mem::zeroed() };
        let mut c: [u8; 256] = [0; 256];
        c['A' as usize] = 0; c['a' as usize] = 0; c['C' as usize] = 1; c['c' as usize] = 1;
        c['G' as usize] = 2; c['g' as usize] = 2; c['T' as usize] = 3; c['t' as usize] = 3; // build the encoding table
        let qs : Vec<u8> = q.iter().map(|x| c[*x as usize]).collect();
        let ts : Vec<u8> = r.iter().map(|x| c[*x as usize]).collect();
        let ql = qs.len() as i32;
        let tl = ts.len() as i32;
        let gapo = 2;
        let gape = 1;

        // preparation for scrooge
        let scrooge_w = 64; // W
        let scrooge_k = 64; // K
        let bitvectors_per_element: usize = 1;
        let scrooge_columns: usize = scrooge_w + 1;
        let scrooge_rows: usize = scrooge_k + 1;
        let scrooge_r_bitvectors: usize = scrooge_columns * scrooge_rows * bitvectors_per_element;
        let mut scrooge_r : Vec<genasm_cpu_halfbitvector> = vec![genasm_cpu_bitvector::default(); scrooge_r_bitvectors];
        let mut scrooge_forefront: Vec<genasm_cpu_bitvector> = vec![genasm_cpu_bitvector::default(); scrooge_w+1];
        let mut scrooge_cigar: Vec<u8> = vec![0; (ql * 4 + 1) as usize];

        // now on to the benchmark
        let mut group = crit.benchmark_group("BenchmarkGroup");
        group.throughput(Throughput::Bytes(len as u64));

        group.bench_with_input(BenchmarkId::new(divname("rust_bio_levenshtein"), len), &(&r,&q), |b: &mut Bencher, i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
            black_box(bio::alignment::distance::levenshtein(i.0, i.1));
        })});

        group.bench_with_input(BenchmarkId::new(divname("parasailors"), len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
            black_box(global_alignment_score(&profile, &r, 2, 1));
        })});

        group.bench_with_input(BenchmarkId::new(divname("rust_bio_simd_bounded_levenshtein"), len), &(&r,&q), |b: &mut Bencher, i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
            black_box(bounded_levenshtein(i.0, i.1, nb_err as u32));
        })});

        group.bench_with_input(BenchmarkId::new(divname("rust_bio_simd_levenshtein"), len), &(&r,&q), |b: &mut Bencher, i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
            black_box(levenshtein(i.0, i.1));
        })});

        group.bench_with_input(BenchmarkId::new(divname("block_aligner"), len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
            let mut block_aligner = Block::<false, false>::new(q_padded.len(), r_padded.len(), block_size);
            block_aligner.align(&q_padded, &r_padded, &NW1, run_gaps, block_size..=block_size, 0);
            let block_score = block_aligner.res().score as u32;
            black_box(block_score);
        })});

        // libwfa doesn't play well with wfa2
        // if one is uncommented, the other needs to be commented,
        // else you get mem corruption for some reason
        /*group.bench_with_input(BenchmarkId::new("libwfa", len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
          let mut libwfa_wavefronts = AffineWavefronts::new_complete(
          q.len(),
          r.len(),
          &mut penalties,
          &libwfa_alloc,
          );*/
        /*
           let mut libwfa_wavefronts = AffineWavefronts::new_reduced(
           q.len(),
           r.len(),
           &mut penalties,
           5,
           25,
           &libwfa_alloc,
           ); 
           let res = libwfa_wavefronts.align(&q, &r);
        //println!("res {:?}",res.res());
        black_box(res);
        })});
        */

        group.bench_with_input(BenchmarkId::new(divname("wfa2"), len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
            let res = wfa2_aligner.align_end_to_end(&q, &r);
            black_box(res);
        })});

        group.bench_with_input(BenchmarkId::new(divname("ksw2_extz"), len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
            unsafe { 
                let res = ksw_extz(std::ptr::null_mut(), ql, qs.as_ptr(), tl, ts.as_ptr(), 5, mat.as_ptr(), gapo, gape, -1, -1, 0, &mut ez);
                black_box(res);
            }
        })});

        group.bench_with_input(BenchmarkId::new(divname("ksw2_extz2_sse"), len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
            unsafe { 
                let res = ksw_extz2_sse(std::ptr::null_mut(), ql, qs.as_ptr(), tl, ts.as_ptr(), 5, mat.as_ptr(), gapo, gape, -1, -1, 10, 0, &mut ez);
                black_box(res);
            }
        })});

        group.bench_with_input(BenchmarkId::new(divname("scrooge"), len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
            unsafe { 
                let res = genasm_cpu_genasm(tl as usize, ts.as_ptr() as *mut i8, ql as usize, qs.as_ptr() as *mut i8, 
                                            scrooge_r.as_mut_ptr(), scrooge_forefront.as_mut_ptr(), scrooge_cigar.as_mut_ptr() as *mut i8);
                black_box(res);
            }
        })});


    group.finish();

    }
}

criterion_group!(benches, lowdivalign_bench);
criterion_main!(benches);
