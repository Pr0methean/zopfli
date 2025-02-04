//! The squeeze functions do enhanced LZ77 compression by optimal parsing with a
//! cost model, rather than greedily choosing the longest length or using a single
//! step of lazy matching like regular implementations.
//!
//! Since the cost model is based on the Huffman tree that can only be calculated
//! after the LZ77 data is generated, there is a chicken and egg problem, and
//! multiple runs are done with updated cost models to converge to a better
//! solution.

use alloc::vec::Vec;
use core::{
    cmp,
    fmt::{Debug, Display, Formatter},
    ops::DerefMut,
};
use std::{iter};
use std::sync::{Arc, Mutex, RwLock};

use genevo::{
    algorithm::EvaluatedPopulation,
    ga,
    genetic::{Children, Genotype, Parents},
    operator::{prelude::ElitistReinserter, CrossoverOp, GeneticOperator, MutationOp, SelectionOp},
    prelude::*,
    random::Rng,
    simulation::State,
    termination::{StopFlag, Termination},
};
use lockfree_object_pool::LinearObjectPool;
use log::{debug, info};
use once_cell::sync::Lazy;
use ordered_float::OrderedFloat;
use rand::{
    distributions::{Bernoulli, Distribution, WeightedIndex},
    seq::SliceRandom,
};
use moka::sync::Cache as MokaCache;

use crate::{
    cache::Cache,
    deflate::{calculate_block_size, BlockType},
    hash::{ZopfliHash, HASH_POOL},
    lz77::{find_longest_match, LitLen, Lz77Store},
    symbols::{get_dist_extra_bits, get_dist_symbol, get_length_extra_bits, get_length_symbol},
    util::{ZOPFLI_MAX_MATCH, ZOPFLI_NUM_D, ZOPFLI_NUM_LL, ZOPFLI_WINDOW_MASK, ZOPFLI_WINDOW_SIZE},
};
use crate::cache::ZopfliLongestMatchCache;
use crate::lz77::ZopfliOutput;

const K_INV_LOG2: f64 = core::f64::consts::LOG2_E; // 1.0 / log(2.0)
const SCORE_CACHE_SIZE: u64 = 16 * 1024;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // False-positive
use crate::math::F64MathExt;

static LZ77_STORE_POOL: Lazy<LinearObjectPool<Lz77Store>> =
    Lazy::new(|| LinearObjectPool::new(Lz77Store::new, Lz77Store::reset));

/// Cost model which should exactly match fixed tree.
const fn get_cost_fixed(litlen: usize, dist: u16) -> f64 {
    let result = if dist == 0 {
        if litlen <= 143 {
            8
        } else {
            9
        }
    } else {
        let dbits = get_dist_extra_bits(dist);
        let lbits = get_length_extra_bits(litlen);
        let lsym = get_length_symbol(litlen);
        // Every dist symbol has length 5.
        7 + (lsym > 279) as usize + 5 + dbits + lbits
    };
    result as f64
}

/// Cost model based on symbol statistics.
fn get_cost_stat(litlen: usize, dist: u16, stats: &SymbolStats) -> f64 {
    if dist == 0 {
        stats.ll_symbols[litlen]
    } else {
        let lsym = get_length_symbol(litlen);
        let lbits = get_length_extra_bits(litlen) as f64;
        let dsym = get_dist_symbol(dist);
        let dbits = get_dist_extra_bits(dist) as f64;
        lbits + dbits + stats.ll_symbols[lsym] + stats.d_symbols[dsym]
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct SymbolTable {
    /* The literal and length symbols. */
    litlens: [usize; ZOPFLI_NUM_LL],
    /* The 32 unique dist symbols, not the 32768 possible dists. */
    dists: [usize; ZOPFLI_NUM_D],
}

impl Default for SymbolTable {
    fn default() -> Self {
        SymbolTable {
            litlens: [0; ZOPFLI_NUM_LL],
            dists: [0; ZOPFLI_NUM_D],
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct SymbolStats {
    table: SymbolTable,
    /* Length of each lit/len symbol in bits. */
    ll_symbols: [f64; ZOPFLI_NUM_LL],
    /* Length of each dist symbol in bits. */
    d_symbols: [f64; ZOPFLI_NUM_D],
}

impl Default for SymbolStats {
    fn default() -> SymbolStats {
        SymbolStats {
            table: Default::default(),
            ll_symbols: [0.0; ZOPFLI_NUM_LL],
            d_symbols: [0.0; ZOPFLI_NUM_D],
        }
    }
}

impl SymbolStats {
    /// Calculates the entropy of each symbol, based on the counts of each symbol. The
    /// result is similar to the result of length_limited_code_lengths, but with the
    /// actual theoritical bit lengths according to the entropy. Since the resulting
    /// values are fractional, they cannot be used to encode the tree specified by
    /// DEFLATE.
    fn calculate_entropy(&mut self) {
        fn calculate_and_store_entropy(count: &[usize], bitlengths: &mut [f64]) {
            let n = count.len();

            let sum = count.iter().sum();

            let log2sum = (if sum == 0 { n } else { sum } as f64).ln() * K_INV_LOG2;

            for i in 0..n {
                // When the count of the symbol is 0, but its cost is requested anyway, it
                // means the symbol will appear at least once anyway, so give it the cost as if
                // its count is 1.
                if count[i] == 0 {
                    bitlengths[i] = log2sum;
                } else {
                    bitlengths[i] = log2sum - (count[i] as f64).ln() * K_INV_LOG2;
                }

                // Depending on compiler and architecture, the above subtraction of two
                // floating point numbers may give a negative result very close to zero
                // instead of zero (e.g. -5.973954e-17 with gcc 4.1.2 on Ubuntu 11.4). Clamp
                // it to zero. These floating point imprecisions do not affect the cost model
                // significantly so this is ok.
                if bitlengths[i] < 0.0 && bitlengths[i] > -1E-5 {
                    bitlengths[i] = 0.0;
                }
                debug_assert!(bitlengths[i] >= 0.0);
            }
        }

        calculate_and_store_entropy(&self.table.litlens, &mut self.ll_symbols);
        calculate_and_store_entropy(&self.table.dists, &mut self.d_symbols);
    }

    /// Appends the symbol statistics from the store.
    fn get_statistics(&mut self, store: &Lz77Store) {
        for &litlen in &store.litlens {
            match litlen {
                LitLen::Literal(lit) => self.table.litlens[lit as usize] += 1,
                LitLen::LengthDist(len, dist) => {
                    self.table.litlens[get_length_symbol(len as usize)] += 1;
                    self.table.dists[get_dist_symbol(dist)] += 1;
                }
            }
        }
        self.table.litlens[256] = 1; /* End symbol. */

        self.calculate_entropy();
    }

    fn clear_freqs(&mut self) {
        self.table.litlens = [0; ZOPFLI_NUM_LL];
        self.table.dists = [0; ZOPFLI_NUM_D];
    }
}

fn add_weighed_stat_freqs(
    stats1: &SymbolStats,
    w1: f64,
    stats2: &SymbolStats,
    w2: f64,
) -> SymbolStats {
    let mut result = SymbolStats::default();

    for i in 0..ZOPFLI_NUM_LL {
        result.table.litlens[i] = (stats1.table.litlens[i] as f64 * w1
            + stats2.table.litlens[i] as f64 * w2
            + 0.5) as usize;
    }
    for i in 0..ZOPFLI_NUM_D {
        result.table.dists[i] =
            (stats1.table.dists[i] as f64 * w1 + stats2.table.dists[i] as f64 * w2 + 0.5) as usize;
    }
    result.table.litlens[256] = 1; // End symbol.
    result
}

/// Finds the minimum possible cost this cost model can return for valid length and
/// distance symbols.
fn get_cost_model_min_cost<F: Fn(usize, u16) -> f64>(costmodel: F) -> f64 {
    let mut bestlength = 0; // length that has lowest cost in the cost model
    let mut bestdist = 0; // distance that has lowest cost in the cost model

    // Table of distances that have a different distance symbol in the deflate
    // specification. Each value is the first distance that has a new symbol. Only
    // different symbols affect the cost model so only these need to be checked.
    // See RFC 1951 section 3.2.5. Compressed blocks (length and distance codes).

    const DSYMBOLS: [u16; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];

    let mut mincost = f64::INFINITY;
    for i in 3..259 {
        let c = costmodel(i, 1);
        if c < mincost {
            bestlength = i;
            mincost = c;
        }
    }

    mincost = f64::INFINITY;
    for dsym in DSYMBOLS {
        let c = costmodel(3, dsym);
        if c < mincost {
            bestdist = dsym;
            mincost = c;
        }
    }
    costmodel(bestlength, bestdist)
}

/// Performs the forward pass for "squeeze". Gets the most optimal length to reach
/// every byte from a previous byte, using cost calculations.
/// `s`: the `ZopfliBlockState`
/// `in_data`: the input data array
/// `instart`: where to start
/// `inend`: where to stop (not inclusive)
/// `costmodel`: function to calculate the cost of some lit/len/dist pair.
/// `length_array`: output array of size `(inend - instart)` which will receive the best
///     length to reach this byte from a previous byte.
/// returns the cost that was, according to the `costmodel`, needed to get to the end.
fn get_best_lengths<F: Fn(usize, u16) -> f64, C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    costmodel: F,
    h: &mut ZopfliHash,
) -> (f64, Vec<u16>) {
    // Best cost to get here so far.
    let blocksize = inend - instart;
    let mut length_array = vec![0; blocksize + 1];
    if instart == inend {
        return (0.0, length_array);
    }
    let windowstart = instart.saturating_sub(ZOPFLI_WINDOW_SIZE);

    h.reset();
    let arr = &in_data[..inend];
    h.warmup(arr, windowstart, inend);
    for i in windowstart..instart {
        h.update(arr, i);
    }
    let mut costs: Vec<f32> = iter::repeat(f32::INFINITY).take(blocksize + 1).collect();
    costs[0] = 0.0; /* Because it's the start. */

    let mut i = instart;
    let mut leng;
    let mut longest_match;
    let mut sublen = vec![0; ZOPFLI_MAX_MATCH + 1];
    let mincost = get_cost_model_min_cost(&costmodel);
    while i < inend {
        let mut j = i - instart; // Index in the costs array and length_array.
        h.update(arr, i);

        // If we're in a long repetition of the same character and have more than
        // ZOPFLI_MAX_MATCH characters before and after our position.
        if h.same[i & ZOPFLI_WINDOW_MASK] > ZOPFLI_MAX_MATCH as u16 * 2
            && i > instart + ZOPFLI_MAX_MATCH + 1
            && i + ZOPFLI_MAX_MATCH * 2 + 1 < inend
            && h.same[(i - ZOPFLI_MAX_MATCH) & ZOPFLI_WINDOW_MASK] > ZOPFLI_MAX_MATCH as u16
        {
            let symbolcost = costmodel(ZOPFLI_MAX_MATCH, 1);
            // Set the length to reach each one to ZOPFLI_MAX_MATCH, and the cost to
            // the cost corresponding to that length. Doing this, we skip
            // ZOPFLI_MAX_MATCH values to avoid calling ZopfliFindLongestMatch.

            for _ in 0..ZOPFLI_MAX_MATCH {
                costs[j + ZOPFLI_MAX_MATCH] = costs[j] + symbolcost as f32;
                length_array[j + ZOPFLI_MAX_MATCH] = ZOPFLI_MAX_MATCH as u16;
                i += 1;
                j += 1;
                h.update(arr, i);
            }
        }

        longest_match = find_longest_match(
            lmc,
            h,
            arr,
            i,
            inend,
            instart,
            ZOPFLI_MAX_MATCH,
            &mut Some(&mut sublen),
        );
        leng = longest_match.length;

        // Literal.
        if i < inend {
            let new_cost = costmodel(arr[i] as usize, 0) + costs[j] as f64;
            debug_assert!(new_cost >= 0.0);
            if new_cost < costs[j + 1] as f64 {
                costs[j + 1] = new_cost as f32;
                length_array[j + 1] = 1;
            }
        }
        // Lengths.
        let kend = cmp::min(leng as usize, inend - i);
        let mincostaddcostj = mincost + costs[j] as f64;

        for (k, &sublength) in sublen.iter().enumerate().take(kend + 1).skip(3) {
            // Calling the cost model is expensive, avoid this if we are already at
            // the minimum possible cost that it can return.
            if costs[j + k] as f64 <= mincostaddcostj {
                continue;
            }

            let new_cost = costmodel(k, sublength) + costs[j] as f64;
            debug_assert!(new_cost >= 0.0);
            if new_cost < costs[j + k] as f64 {
                debug_assert!(k <= ZOPFLI_MAX_MATCH);
                costs[j + k] = new_cost as f32;
                length_array[j + k] = k as u16;
            }
        }
        i += 1;
    }

    debug_assert!(costs[blocksize] >= 0.0);
    (costs[blocksize] as f64, length_array)
}

/// Calculates the optimal path of lz77 lengths to use, from the calculated
/// `length_array`. The `length_array` must contain the optimal length to reach that
/// byte. The path will be filled with the lengths to use, so its data size will be
/// the amount of lz77 symbols.
fn trace(size: usize, length_array: &[u16]) -> Vec<u16> {
    let mut index = size;
    if size == 0 {
        return vec![];
    }
    let mut path = Vec::with_capacity(index);

    while index > 0 {
        let lai = length_array[index];
        let laiu = lai as usize;
        path.push(lai);
        debug_assert!(laiu <= index);
        debug_assert!(laiu <= ZOPFLI_MAX_MATCH);
        debug_assert_ne!(lai, 0);
        index -= laiu;
    }

    path
}

/// Does a single run for `lz77_optimal`. For good compression, repeated runs
/// with updated statistics should be performed.
/// `s`: the block state
/// `in_data`: the input data array
/// `instart`: where to start
/// `inend`: where to stop (not inclusive)
/// `length_array`: array of size `(inend - instart)` used to store lengths
/// `costmodel`: function to use as the cost model for this squeeze run
/// `store`: place to output the LZ77 data
/// returns the cost that was, according to the `costmodel`, needed to get to the end.
///     This is not the actual cost.
#[allow(clippy::too_many_arguments)] // Not feasible to refactor in a more readable way
fn lz77_optimal_run<F: Fn(usize, u16) -> f64, C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    costmodel: F,
    store: &mut Lz77Store,
    h: &mut ZopfliHash,
) {
    let (cost, length_array) = get_best_lengths(lmc, in_data, instart, inend, costmodel, h);
    let path = trace(inend - instart, &length_array);
    store.follow_path(in_data, instart, inend, path, lmc);
    debug_assert!(cost < f64::INFINITY);
}

/// Does the same as `lz77_optimal`, but optimized for the fixed tree of the
/// deflate standard.
/// The fixed tree never gives the best compression. But this gives the best
/// possible LZ77 encoding possible with the fixed tree.
/// This does not create or output any fixed tree, only LZ77 data optimized for
/// using with a fixed tree.
/// If `instart` is larger than `0`, it uses values before `instart` as starting
/// dictionary.
pub fn lz77_optimal_fixed<C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    store: &mut Lz77Store,
) {
    let mut h = ZopfliHash::new();
    lz77_optimal_run(lmc, in_data, instart, inend, get_cost_fixed, store, &mut h);
}

impl Genotype for SymbolTable {
    type Dna = usize;
}

#[derive(Copy, Clone, Debug)]
struct SymbolTableMutator<T>
where
    T: Distribution<bool>,
{
    mutation_chance_distro: T,
    mutate_by_copying_chance_distro: T,
    max_litlen_freq: usize,
    max_dist_freq: usize,
}

impl<T> GeneticOperator for SymbolTableMutator<T>
where
    T: Distribution<bool> + Clone,
{
    fn name() -> String {
        "SymbolTableMutator".to_string()
    }
}

impl<T> MutationOp<SymbolTable> for SymbolTableMutator<T>
where
    T: Distribution<bool> + Clone,
{
    fn mutate<R>(&self, mut genome: SymbolTable, rng: &mut R) -> SymbolTable
    where
        R: Rng + Sized,
    {
        (0..ZOPFLI_NUM_LL).for_each(|index| {
            if index != 256 {
                // don't mutate the end symbol
                mutate_gene(
                    &self.mutation_chance_distro,
                    &self.mutate_by_copying_chance_distro,
                    0,
                    self.max_litlen_freq,
                    rng,
                    index,
                    &mut genome.litlens,
                );
            }
        });
        (0..ZOPFLI_NUM_D).for_each(|index| {
            mutate_gene(
                &self.mutation_chance_distro,
                &self.mutate_by_copying_chance_distro,
                0,
                self.max_dist_freq,
                rng,
                index,
                &mut genome.dists,
            );
        });
        genome
    }
}

fn mutate_gene<R, D, E>(
    mutation_chance_distribution: &D,
    mutate_by_copying_chance_distribution: &E,
    min_value: usize,
    max_value: usize,
    rng: &mut R,
    index: usize,
    chromosome: &mut [usize],
) where
    R: Rng + Sized,
    D: Distribution<bool>,
    E: Distribution<bool>,
{
    if mutation_chance_distribution.sample(rng) {
        let new_allele = if max_value - min_value > 1
            && mutate_by_copying_chance_distribution.sample(rng) {
            chromosome[rng.gen_range(0..chromosome.len())]
        } else {
            rng.gen_range(min_value..=max_value)
        };
        chromosome[index] = new_allele;
    }
}

impl From<SymbolTable> for SymbolStats {
    fn from(value: SymbolTable) -> Self {
        let mut stats = SymbolStats {
            table: value,
            ..Default::default()
        };
        stats.calculate_entropy();
        stats
    }
}

impl Phenotype<SymbolTable> for SymbolStats {
    fn genes(&self) -> SymbolTable {
        self.table
    }

    fn derive(&self, new_genes: SymbolTable) -> Self {
        SymbolStats::from(new_genes)
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
struct FloatAsFitness(OrderedFloat<f64>);

impl Display for FloatAsFitness {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl From<f64> for FloatAsFitness {
    fn from(value: f64) -> Self {
        FloatAsFitness(value.into())
    }
}

impl From<FloatAsFitness> for f64 {
    fn from(value: FloatAsFitness) -> f64 {
        value.0.into()
    }
}

impl Fitness for FloatAsFitness {
    fn zero() -> Self {
        0.0.into()
    }

    fn abs_diff(&self, other: &Self) -> Self {
        f64::abs(self.0 .0 - other.0 .0).into()
    }
}

#[derive(Debug)]
struct ZopfliGaState<'a> {
    pub best: RwLock<ZopfliOutput>,
    pub score_cache: Arc<MokaCache<SymbolTable, f64>>,
    pub lmc: Arc<Mutex<ZopfliLongestMatchCache>>,
    pub data: &'a [u8],
    pub blockstart: usize,
    pub blockend: usize
}

impl <'a> Clone for ZopfliGaState<'a> {
    fn clone(&self) -> Self {
        ZopfliGaState {
            best: RwLock::new(self.best.read().unwrap().clone()),
            score_cache: self.score_cache.clone(),
            lmc: self.lmc.clone(),
            ..*self
        }
    }
}

impl <'a> FitnessFunction<SymbolTable, FloatAsFitness> for &'a ZopfliGaState<'a> {
    fn fitness_of(&self, a: &SymbolTable) -> FloatAsFitness {
        self.score_cache
            .get_with(*a, || {
                let read_best = self.best.read().unwrap();
                if read_best.stats == *a {
                    return (-read_best.cost).into();
                }
                let best_before = read_best.cost;
                drop(read_best);
                let stats = SymbolStats::from(*a);
                let pool = &*LZ77_STORE_POOL;
                let mut currentstore = pool.pull();
                let mut lmc = self.lmc.lock().unwrap();
                lz77_optimal_run(
                    lmc.deref_mut(),
                    self.data,
                    self.blockstart,
                    self.blockend,
                    |a, b| get_cost_stat(a, b, &stats),
                    currentstore.deref_mut(),
                    &mut ZopfliHash::new(),
                );
                lz77_deterministic_loop(
                    lmc.deref_mut(),
                    self.data,
                    self.blockstart,
                    self.blockend,
                    currentstore.deref_mut()
                );
                let cost =
                    calculate_block_size(&currentstore, 0, currentstore.size(), BlockType::Dynamic);
                if cost < best_before {
                    let mut best = self.best.write().unwrap();
                    let best = best.deref_mut();
                    if cost < best.cost {
                        *best = ZopfliOutput {
                            stored: currentstore.clone(),
                            stats: stats.table,
                            cost,
                        };
                    }
                }
                -cost
            })
            .into()
    }

    fn average(&self, a: &[FloatAsFitness]) -> FloatAsFitness {
        let mut total = 0.0;
        a.iter().for_each(|value| total += value.0 .0);
        (total / (a.len() as f64)).into()
    }

    fn highest_possible_fitness(&self) -> FloatAsFitness {
        (-8.0).into()
    }

    fn lowest_possible_fitness(&self) -> FloatAsFitness {
        f64::NEG_INFINITY.into()
    }
}

#[derive(Debug)]
pub struct SymbolTableBuilder {
    first_guess: SymbolTable,
    fixed_population: Vec<SymbolTable>,
    max_litlen_freq: usize,
    max_dist_freq: usize,
}

impl SymbolTableBuilder {
    fn new(first_guess: SymbolTable, second_guess: SymbolTable,
           mut max_litlen_freq: usize, mut max_dist_freq: usize) -> Self {
        if max_litlen_freq == 0 {
            max_litlen_freq = 1;
        }
        if max_dist_freq == 0 {
            max_dist_freq = 1;
        }
        let mut fixed_litlens = Vec::with_capacity(8);
        let mut fixed_dists = Vec::with_capacity(8);
        fixed_litlens.push(first_guess.litlens);
        fixed_dists.push(first_guess.dists);

        let mut sorted_litlens = first_guess.litlens;
        sorted_litlens.sort_unstable();
        sorted_litlens.reverse();
        sorted_litlens[256] = 1; // End symbol
        fixed_litlens.push(sorted_litlens);
        if second_guess.litlens != first_guess.litlens {
            fixed_litlens.push(second_guess.litlens);
            let mut sorted_litlens = second_guess.litlens;
            sorted_litlens.sort_unstable();
            sorted_litlens.reverse();
            sorted_litlens[256] = 1; // End symbol
            fixed_litlens.push(sorted_litlens);
        }
        if second_guess.dists != first_guess.dists {
            fixed_dists.push(second_guess.dists);
        }
        if max_dist_freq > 1 {
            let mut sorted_dists = first_guess.dists;
            fixed_dists.push(sorted_dists);
            sorted_dists.sort_unstable();
            sorted_dists.reverse();
            let mut sorted_dists = sorted_dists.into_iter();
            let mut nonzero_sorted_dists = first_guess.dists.clone();
            for dist in nonzero_sorted_dists.iter_mut() {
                if *dist != 0 {
                    *dist = sorted_dists.next().unwrap();
                }
            }
            fixed_dists.push(nonzero_sorted_dists);
            if second_guess.dists != first_guess.dists {
                let mut sorted_dists = second_guess.dists;
                sorted_dists.sort_unstable();
                let mut mostly_sorted_dists = sorted_dists;

                // dists[0] almost always ends up with the largest value
                mostly_sorted_dists[0] = mostly_sorted_dists[ZOPFLI_NUM_D - 1];

                fixed_dists.push(mostly_sorted_dists);
                sorted_dists.reverse();
                fixed_dists.push(sorted_dists.iter().copied().collect::<Vec<_>>().try_into().unwrap());
                let mut sorted_dists = sorted_dists.into_iter();
                let mut nonzero_sorted_dists = second_guess.dists.clone();
                for dist in nonzero_sorted_dists.iter_mut() {
                    if *dist != 0 {
                        *dist = sorted_dists.next().unwrap();
                    }
                }
                fixed_dists.push(nonzero_sorted_dists);
            }
        } else {
            let mut sorted_dists = first_guess.dists;
            sorted_dists.sort();
            let mut mostly_sorted_dists = sorted_dists;

            // dists[0] almost always ends up with the largest value
            mostly_sorted_dists[0] = mostly_sorted_dists[ZOPFLI_NUM_D - 1];

            fixed_dists.push(mostly_sorted_dists);
            sorted_dists.reverse();
            fixed_dists.push(sorted_dists);
        }
        let mut zero_litlens = [0; ZOPFLI_NUM_LL];
        zero_litlens[256] = 1;
        fixed_litlens.push(zero_litlens);
        if max_litlen_freq > 0 {
            let mut maxed_litlens = [max_litlen_freq; ZOPFLI_NUM_LL];
            maxed_litlens[256] = 1;
            fixed_litlens.push(maxed_litlens);
            if max_litlen_freq > 1 {
                let litlens_ones = [1; ZOPFLI_NUM_LL];
                fixed_litlens.push(litlens_ones);
            }
        }
        let zero_dists = [0; ZOPFLI_NUM_D];
        fixed_dists.push(zero_dists);
        if max_dist_freq > 0 {
            let maxed_dists = [max_dist_freq; ZOPFLI_NUM_D];
            fixed_dists.push(maxed_dists);
            if max_dist_freq > 1 {
                let dists_ones = [1; ZOPFLI_NUM_D];
                fixed_dists.push(dists_ones);
            }
        }
        let fixed_population: Vec<_> = fixed_litlens
            .into_iter()
            .flat_map(|litlens| {
                fixed_dists.iter().map(move |dists| SymbolTable {
                    litlens,
                    dists: *dists,
                })
            })
            .collect();
        info!("Fixed population size is {}", fixed_population.len());
        SymbolTableBuilder {
            first_guess,
            max_litlen_freq,
            max_dist_freq,
            fixed_population,
        }
    }
}

impl GenomeBuilder<SymbolTable> for SymbolTableBuilder {
    fn build_genome<R>(&self, index: usize, rng: &mut R) -> SymbolTable
    where
        R: Rng + Sized,
    {
        if index < self.fixed_population.len() {
            self.fixed_population[index]
        } else {
            if index % 4 != 0 {
                let mut table = SymbolTable::default();
                for litlen in table.litlens.iter_mut() {
                    *litlen = rng.gen_range(0..=self.max_litlen_freq);
                }
                table.litlens[256] = 1; // end symbol
                for dist in table.dists.iter_mut() {
                    if self.max_dist_freq == 1 || index % 2 == 0 || *dist != 0 {
                        *dist = rng.gen_range(0..=self.max_dist_freq);
                    }
                }
                table
            } else {
                let mut table = self.first_guess;
                table.litlens.shuffle(rng);
                table.litlens[256] = 1; // end symbol
                table.dists.shuffle(rng);
                table
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct RankBasedSelector {
    selection_ratio: f64,
    num_individuals_per_parents: usize,
}

impl GeneticOperator for RankBasedSelector {
    fn name() -> String {
        "RankBasedSelector".to_string()
    }
}

impl<G, F> SelectionOp<G, F> for RankBasedSelector
where
    G: Genotype,
    F: Fitness,
{
    fn select_from<R>(&self, population: &EvaluatedPopulation<G, F>, rng: &mut R) -> Vec<Parents<G>>
    where
        R: Rng + Sized,
    {
        let individuals = population.individuals().to_vec();
        let fitness = population.fitness_values();
        let mut ranked: Vec<_> = individuals
            .into_iter()
            .zip(fitness.iter().cloned())
            .collect();
        ranked.sort_unstable_by_key(|(_, fitness)| fitness.clone());
        let dist = WeightedIndex::new(1..=ranked.len()).unwrap();
        let num_parents = (ranked.len() as f64 * self.selection_ratio + 0.5) as usize;
        let mut parents = Vec::with_capacity(num_parents);
        for _ in 0..num_parents {
            let mut parent_indices = Vec::with_capacity(self.num_individuals_per_parents);
            while parent_indices.len() < self.num_individuals_per_parents {
                parent_indices.push(dist.sample(rng));
                parent_indices.sort();
                parent_indices.dedup();
            }
            parent_indices.shuffle(rng);
            parents.push(
                parent_indices
                    .into_iter()
                    .map(|index| ranked[index].0.clone())
                    .collect(),
            );
        }
        parents
    }
}

#[derive(Debug)]
struct GenerationsWithoutImprovementLimiter {
    current_best: f64,
    generations_without_improvement: u64,
    max_generations_without_improvement: u64,
    generations: u64,
    max_generations: u64,
}

impl GenerationsWithoutImprovementLimiter {
    fn new(max_generations_without_improvement: u64, max_generations: u64) -> Self {
        GenerationsWithoutImprovementLimiter {
            current_best: f64::NEG_INFINITY,
            generations_without_improvement: 0,
            max_generations_without_improvement,
            generations: 0,
            max_generations,
        }
    }
}

impl<A, G> Termination<A> for GenerationsWithoutImprovementLimiter
where
    A: Algorithm<Output = ga::State<G, FloatAsFitness>>,
    G: Genotype,
{
    fn evaluate(&mut self, state: &State<A>) -> StopFlag {
        self.generations += 1;
        if self.generations >= self.max_generations
        {
            StopFlag::StopNow("Maximum generations reached".into())
        } else {
            let fitness = state.result.best_solution.solution.fitness.0 .0;
            if fitness >= -8.0 {
                StopFlag::StopNow("Already only 1 byte".into())
            } else if fitness > self.current_best {
                self.current_best = fitness;
                self.generations_without_improvement = 0;
                StopFlag::Continue
            } else {
                self.generations_without_improvement += 1;
                if self.generations_without_improvement >= self.max_generations_without_improvement
                {
                    StopFlag::StopNow("Generations-without-improvement limit reached".into())
                } else {
                    StopFlag::Continue
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct SymbolTableCrossBreeder<D>
where
    D: Distribution<bool> + Clone,
{
    crossover_chance_dist: D,
}

impl<D> GeneticOperator for SymbolTableCrossBreeder<D>
where
    D: Distribution<bool> + Clone,
{
    fn name() -> String {
        "SymbolTableCrossBreeder".to_string()
    }
}

fn generate_child_chromosomes<const N: usize, R>(
    parent0: [usize; N],
    parent1: [usize; N],
    rng: &mut R,
) -> [[usize; N]; 2]
where
    R: Rng + Sized,
{
    let cut_point = rng.gen_range(0..=N);
    let mut hybrid_0 = [0; N];
    hybrid_0[..cut_point].copy_from_slice(&parent0[..cut_point]);
    hybrid_0[cut_point..].copy_from_slice(&parent1[cut_point..]);
    let mut hybrid_1 = [0; N];
    hybrid_1[..cut_point].copy_from_slice(&parent1[..cut_point]);
    hybrid_1[cut_point..].copy_from_slice(&parent0[cut_point..]);
    [hybrid_0, hybrid_1]
}

impl<D> CrossoverOp<SymbolTable> for SymbolTableCrossBreeder<D>
where
    D: Distribution<bool> + Clone,
{
    fn crossover<R>(&self, parents: Parents<SymbolTable>, rng: &mut R) -> Children<SymbolTable>
    where
        R: Rng + Sized,
    {
        let num_parents = parents.len();
        if num_parents < 2 {
            return vec![];
        }
        let mut children = Vec::with_capacity(num_parents * (num_parents - 1));
        for first_parent_index in 0..num_parents - 1 {
            let first_parent = &parents[first_parent_index];
            for second_parent_index in first_parent_index + 1..num_parents {
                let second_parent = &parents[second_parent_index];
                if self.crossover_chance_dist.sample(rng) {
                    let litlens = generate_child_chromosomes(
                        first_parent.litlens,
                        second_parent.litlens,
                        rng,
                    );
                    let dists =
                        generate_child_chromosomes(second_parent.dists, first_parent.dists, rng);
                    if rng.gen_bool(0.5) {
                        children.push(SymbolTable {
                            litlens: litlens[0],
                            dists: dists[1],
                        });
                        children.push(SymbolTable {
                            litlens: litlens[1],
                            dists: dists[0],
                        });
                    } else {
                        children.push(SymbolTable {
                            litlens: litlens[0],
                            dists: dists[0],
                        });
                        children.push(SymbolTable {
                            litlens: litlens[1],
                            dists: dists[1],
                        });
                    }
                } else {
                    children.push(SymbolTable {
                        litlens: first_parent.litlens,
                        dists: second_parent.dists,
                    });
                    children.push(SymbolTable {
                        litlens: second_parent.litlens,
                        dists: first_parent.dists,
                    });
                }
            }
        }
        children
    }
}

/// Calculates lit/len and dist pairs for given data.
/// If `instart` is larger than 0, it uses values before `instart` as starting
/// dictionary.
pub fn lz77_optimal(
    in_data: &[u8],
    instart: usize,
    inend: usize,
    max_iterations: u64,
    max_iterations_without_improvement: u64,
) -> Lz77Store {
    const POPULATION_SIZE: usize = 256;
    const SELECTION_RATIO: f64 = 0.35;
    const NUM_INDIVIDUALS_PER_PARENT: usize = 2;
    const MUTATION_RATE: f64 = 0.01;
    const MUTATION_CHANCE_DIST: Lazy<Bernoulli> =
        Lazy::new(|| Bernoulli::new(MUTATION_RATE).unwrap());
    const REPLACE_RATIO: f64 = 0.7;
    const CROSSOVER_CHANCE: f64 = 0.6;
    const MUTATE_BY_COPYING_CHANCE: f64 = 0.5;
    const CROSSOVER_DIST: Lazy<Bernoulli> = Lazy::new(|| Bernoulli::new(CROSSOVER_CHANCE).unwrap());
    const MUTATE_BY_COPYING_DIST: Lazy<Bernoulli> =
        Lazy::new(|| Bernoulli::new(MUTATE_BY_COPYING_CHANCE).unwrap());

    /* Dist to get to here with smallest cost. */
    let mut outputstore = Lz77Store::new();

    /* Initial run. */
    let mut lmc = ZopfliLongestMatchCache::new(inend - instart);
    outputstore.greedy(&mut lmc, in_data, instart, inend);
    let mut greedy_stats = SymbolStats::default();
    greedy_stats.get_statistics(&outputstore);
    let best_before_ga = lz77_deterministic_loop(
        &mut lmc,
        in_data,
        instart,
        inend,
        &mut outputstore,
    );
    let best_stats_before_ga = best_before_ga.stats;
    debug!(
        "Symbol table at start of GA run: {:?}",
        best_stats_before_ga
    );
    let max_litlen_freq = *best_before_ga.stats.litlens.iter().chain(greedy_stats.table.litlens.iter()).max().unwrap();
    let max_dist_freq = *best_before_ga.stats.dists.iter().chain(greedy_stats.table.dists.iter()).max().unwrap();
    let genome_builder =
        SymbolTableBuilder::new(best_stats_before_ga, greedy_stats.table, max_dist_freq, max_litlen_freq);
    let score_cache = MokaCache::new(SCORE_CACHE_SIZE);
    let mut prev_best = -best_before_ga.cost;
    // score_cache.insert(best_before_ga.stats, prev_best);
    let s = ZopfliGaState {
        best: RwLock::new(best_before_ga).into(),
        score_cache: score_cache.into(),
        lmc: Arc::new(Mutex::new(lmc)),
        data: &in_data,
        blockstart: instart,
        blockend: inend
    };
    let initial_population = build_population()
        .with_genome_builder(genome_builder)
        .of_size(POPULATION_SIZE)
        .uniform_at_random();
    let algorithm = genetic_algorithm()
        .with_evaluation(&s)
        .with_selection(RankBasedSelector {
            selection_ratio: SELECTION_RATIO,
            num_individuals_per_parents: NUM_INDIVIDUALS_PER_PARENT,
        })
        .with_crossover(SymbolTableCrossBreeder {
            crossover_chance_dist: *CROSSOVER_DIST,
        })
        .with_mutation(SymbolTableMutator {
            mutation_chance_distro: *MUTATION_CHANCE_DIST,
            mutate_by_copying_chance_distro: *MUTATE_BY_COPYING_DIST,
            max_litlen_freq,
            max_dist_freq,
        })
        .with_reinsertion(ElitistReinserter::new(&s, false, REPLACE_RATIO))
        .with_initial_population(initial_population)
        .build();
    let mut genetic_algorithm_sim = simulate(algorithm)
        .until(GenerationsWithoutImprovementLimiter::new(
            max_iterations_without_improvement,
            max_iterations,
        ))
        .build();
    loop {
        match genetic_algorithm_sim.step() {
            Ok(SimResult::Intermediate(step)) => {
                let best_solution = step.result.best_solution;
                if best_solution.solution.fitness.0 .0 > prev_best {
                    let evaluated_population = step.result.evaluated_population;
                    prev_best = best_solution.solution.fitness.0 .0;
                    debug!(
                        "step: generation: {}, average_fitness: {:.3}, \
                         best fitness: {}, duration: {}, processing_time: {}",
                        step.iteration,
                        evaluated_population.average_fitness(),
                        best_solution.solution.fitness,
                        step.duration,
                        step.processing_time,
                    );
                }
            }
            Ok(SimResult::Final(step, processing_time, duration, stop_reason)) => {
                debug!(
                    "Result at end of GA: generation: {},\
                         best fitness: {}, symbol table: {:?}, duration: {}, processing_time: {}, stop_reason: {}",
                    step.iteration,
                    step.result.best_solution.solution.fitness,
                    step.result.best_solution.solution.genome,
                    duration,
                    processing_time,
                    stop_reason
                );
                let mut best_after_ga = s.best.read().unwrap().stored.clone();
                let mut best_stats_after_ga = SymbolStats::default();
                best_stats_after_ga.get_statistics(&best_after_ga);
                if best_stats_after_ga.table != best_stats_before_ga {
                    lz77_deterministic_loop(
                        s.lmc.lock().unwrap().deref_mut(),
                        in_data,
                        instart,
                        inend,
                        &mut best_after_ga,
                    );
                }
                debug!("Block finished");
                return best_after_ga;
            }
            Err(e) => panic!("{:?}", e),
        }
    }
}

fn lz77_deterministic_loop<C: Cache>(
    lmc: &mut C,
    in_data: &[u8],
    instart: usize,
    inend: usize,
    outputstore: &mut Lz77Store,
) -> ZopfliOutput {
    let mut lz77_store = LZ77_STORE_POOL.pull();
    let current_store = lz77_store.deref_mut();
    let mut best_cost =
        calculate_block_size(&outputstore, 0, outputstore.size(), BlockType::Dynamic);
    let mut last_cost = best_cost;
    let mut stats = SymbolStats::default();
    stats.get_statistics(&outputstore);
    current_store.clone_from(&outputstore);
    let mut best_stats = stats;
    let hash_pool = &*HASH_POOL;
    let mut h = hash_pool.pull();
    loop {
        let last_stats = stats;
        lz77_optimal_run(
            lmc,
            in_data,
            instart,
            inend,
            |a, b| get_cost_stat(a, b, &stats),
            current_store,
            &mut h,
        );
        let cost =
            calculate_block_size(&current_store, 0, current_store.size(), BlockType::Dynamic);
        stats.clear_freqs();
        stats.get_statistics(&current_store);
        if cost < best_cost {
            debug!("Reduced cost to {} deterministically", cost);
            best_cost = cost;
            stats.calculate_entropy();
            best_stats = stats;
            outputstore.clone_from(&current_store);
        }
        if cost >= last_cost - f64::EPSILON {
            break;
        }
        stats = add_weighed_stat_freqs(&stats, 2.0 / 3.0, &last_stats, 1.0 / 3.0);
        stats.calculate_entropy();
        last_cost = cost;
        current_store.reset();
    }
    ZopfliOutput {
        stored: outputstore.clone(),
        stats: best_stats.table,
        cost: best_cost,
    }
}
