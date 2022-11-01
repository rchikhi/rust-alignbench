#[macro_use]
extern crate criterion;


use rand::prelude::*;
use criterion::{Bencher, Criterion, Throughput, BenchmarkId, black_box};

use block_aligner::simulate::*;
use block_aligner::scan_block::*;
use block_aligner::scores::*;

use bio::alignment::distance::simd::{bounded_levenshtein, levenshtein};

use parasailors::{Matrix, *};

use libwfa::{affine_wavefront::*, bindings::*, penalties::*};

use rust_wfa2::aligner::*;

fn lowdivalign_bench(c: &mut Criterion) {

    let divergence = 0.07;

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


    let mut group = c.benchmark_group("BenchmarkGroup");
    group.throughput(Throughput::Bytes(len as u64));

    group.bench_with_input(BenchmarkId::new("rust_bio_levenshtein", len), &(&r,&q), |b: &mut Bencher, i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
        let res = bio::alignment::distance::levenshtein(i.0, i.1);
        //println!("res {:?}",res);
        black_box(res);
    })});

    group.bench_with_input(BenchmarkId::new("parasailors", len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
        let res = global_alignment_score(&profile, &r, 2, 1);
        //println!("res {:?}",res.res());
        black_box(res);
    })});

    group.bench_with_input(BenchmarkId::new("rust_bio_simd_bounded_levenshtein", len), &(&r,&q), |b: &mut Bencher, i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
        let res = bounded_levenshtein(i.0, i.1, nb_err as u32);
        //println!("res {:?}",res);
        black_box(res);
    })});

    group.bench_with_input(BenchmarkId::new("rust_bio_simd_levenshtein", len), &(&r,&q), |b: &mut Bencher, i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
        let res = levenshtein(i.0, i.1);
        //println!("res {:?}",res);
        black_box(res);
    })});

    group.bench_with_input(BenchmarkId::new("block_aligner", len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
        let res = Block::<_, false, false>::align(&q_padded, &r_padded, &NW1, run_gaps, block_size..=block_size, 0);
        //println!("res {:?}",res.res());
        black_box(res);
    })});

    // libwfa doesn't play well with wfa2
    // if one is uncommented, the other needs to be commented,
    // else you get mem corruption for some reason
    group.bench_with_input(BenchmarkId::new("libwfa", len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
        /*let mut libwfa_wavefronts = AffineWavefronts::new_complete(
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
        */
    })});

    group.bench_with_input(BenchmarkId::new("wfa2", len), &(&r,&q), |b: &mut Bencher, _i: &(&Vec<u8>,&Vec<u8>)| { b.iter(|| {
        let mut wfa2_aligner = WFAlignerGapAffine::new(4, 6, 2, AlignmentScope::Alignment, MemoryModel::MemoryHigh);
        wfa2_aligner.set_heuristic(Heuristic::BandedAdaptive(-10, 10, 1));
        let res = wfa2_aligner.align_end_to_end(&q, &r);
        //println!("res {:?}",res.res());
        black_box(res);
    })});


    group.finish();
}

criterion_group!(benches, lowdivalign_bench);
criterion_main!(benches);
