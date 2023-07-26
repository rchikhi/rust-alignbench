use rand::prelude::*;

use simulate_seqs::*;
use block_aligner::scan_block::*;
use block_aligner::scores::*;

use bio::alignment::distance::simd::{bounded_levenshtein, levenshtein};

use parasailors::{Matrix, *};

//use libwfa::{affine_wavefront::*, bindings::*, penalties::*};

use rust_wfa2::aligner::*;

use ksw2_sys::*;

use Scrooge_sys::*;

#[test]
fn tests() {

    let divergence = 0.07;

    let mut rng = StdRng:: from_entropy();

    // simulate q and r
    let len = 1000;
    let nb_err = ((divergence as f64) * (len as f64)) as usize;
    let r = rand_str(len, &NUC, &mut rng);
    let q = rand_mutate(&r, nb_err, &NUC, &mut rng);
     
    println!("aligning:\nr = {}\nq = {}", String::from_utf8(r.clone()).unwrap(), String::from_utf8(q.clone()).unwrap());
    println!("sequence lengths: {} {}", r.len(), q.len());

    // preparation for block_aligner
    let block_size = 16;
    let run_gaps = Gaps { open: -2, extend: -1 };
    let r_padded = PaddedBytes::from_bytes::<NucMatrix>(&r, block_size);
    let q_padded = PaddedBytes::from_bytes::<NucMatrix>(&q, block_size);

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

    // preparation for ksw2
    let a = 1;
    let b = -1;
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


    // rust-bio standard ED
    let bio_res = bio::alignment::distance::levenshtein(&r,&q);

    // rust-bio standard ED, bounded
    let bio_res2 = bounded_levenshtein(&r,&q, nb_err as u32).unwrap();
    assert!(bio_res == bio_res2);

    // rust-bio SIMD
    let bio_res3 = levenshtein(&r,&q);
    assert!(bio_res == bio_res3);
    println!("rust-bio edit distance: {}", bio_res3);


    // parasail
    let para_res = global_alignment_score(&profile, &r, 2, 1) as u32;
    println!("parasail similarity: {}", para_res);
    // parasail computes NW with fixed matrix (Identity or IdentityWithPenalty), 
    // can't have it output edit distance
    // parasailors also doesn't support outputting cigar


    // block-aligner
    let mut block_aligner = Block::<true, false>::new(q_padded.len(), r_padded.len(), block_size);
    block_aligner.align(&q_padded, &r_padded, &NW1, run_gaps, block_size..=block_size, 0);
    let block_score = block_aligner.res().score as u32;
    println!("block_aligner similarity: {}", block_score);
    let mut block_cigar = block_aligner::cigar::Cigar::new(q_padded.len() as usize, r_padded.len() as usize);
    block_aligner.trace().cigar(q_padded.len(), r_padded.len(), &mut block_cigar);
    println!("block_aligner cigar:\n{}", block_cigar.to_string());
    assert!(para_res == block_score);


    // wfa2 ED
    //let mut wfa2_aligner = WFAlignerGapAffine::new(1, 2, 1, AlignmentScope::Alignment, MemoryModel::MemoryHigh);
    let mut wfa2_aligner = WFAlignerEdit::new(AlignmentScope::Alignment, MemoryModel::MemoryHigh);
    wfa2_aligner.set_heuristic(Heuristic::None);//BandedAdaptive(-10, 10, 1));
    let _wfa2_res = wfa2_aligner.align_end_to_end(&q, &r);
    println!("wfa2 ed: {}", wfa2_aligner.score());
    assert!(wfa2_aligner.score() as u32 == bio_res3);

    // wfa2 NW
    let mut wfa2_aligner = WFAlignerEdit::new(AlignmentScope::Alignment, MemoryModel::MemoryHigh);
    wfa2_aligner.set_heuristic(Heuristic::None);//BandedAdaptive(-10, 10, 1));
    let _wfa2_res = wfa2_aligner.align_end_to_end(&q, &r);
    println!("wfa2 ed: {}", wfa2_aligner.score());
    assert!(wfa2_aligner.score() as u32 == bio_res3);



    // ksw2
    unsafe { 
        let res = ksw_extz(std::ptr::null_mut(), ql, qs.as_ptr(), tl, ts.as_ptr(), 5, mat.as_ptr(), gapo, gape, -1, -1, 0, &mut ez);
        println!("ksw2 similarity: {}", ez.score);
        println!("ksw2 cigar:");
        let cigar = std::slice::from_raw_parts_mut(ez.cigar,ez.n_cigar as usize);
        for i in 0..(ez.n_cigar as usize) {
            print!("{}{}", cigar[i]>>4, "MID".chars().nth((cigar[i]&0xf) as usize).unwrap());
        }
        println!("");
    }

    // scrooge
    unsafe { 
        let res = genasm_cpu_genasm(tl as usize, ts.as_ptr() as *mut i8, ql as usize, qs.as_ptr() as *mut i8, 
                                    scrooge_r.as_mut_ptr(), scrooge_forefront.as_mut_ptr(), scrooge_cigar.as_mut_ptr() as *mut i8);
        println!("scrooge ed: {}", res);
    }

}
