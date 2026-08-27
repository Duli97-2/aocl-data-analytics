/* ************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef NEAREST_NEIGHBORS_HEURISTIC_HPP
#define NEAREST_NEIGHBORS_HEURISTIC_HPP

/*
 * Automatic algorithm selection for radius-neighbour search.
 *
 * ---------------------------------------------------------------------------
 * MODEL
 * ---------------------------------------------------------------------------
 * SNN prunes by projecting the index onto its leading principal direction and
 * keeping only points whose projection lies within R of the query's. The share
 * of the index that survives is the BAND FRACTION; the speedup ceiling is
 * 1/band. Writing w for the cost of one distance computation and h for SNN's
 * fixed per-query overhead (binary search, gather, gemm setup):
 *
 *     brute cost per query  = w * n
 *     snn   cost per query  = h + w * (band * n)
 *
 * Dividing gives the predicted speedup, with c0 = h/w expressed as a number of
 * distance computations:
 *
 *     speedup = n / (band * n + c0(d))          c0(d) = C0_A * d^C0_B
 *
 * Two consequences drive the rule below. There is a break-even band of
 * 1 - c0/n above which SNN cannot win, and a maximum attainable speedup of
 * n/c0 no matter how well the data prunes -- so a dataset that prunes to a
 * handful of candidates can still lose, because there is nothing left for the
 * distance kernel to amortise its setup against.
 *
 * Band itself is modelled without a PCA. If the projections were Gaussian,
 * band = erf(R / 2*sigma_p); substituting the relationship between sigma_p and
 * the pairwise distance scale, and fitting the residual exponent, gives
 *
 *     band ~ erf( BAND_A * (R / median_pairwise_distance)^BAND_B )
 *
 * which needs only a sampled block of distances.
 *
 * ---------------------------------------------------------------------------
 * PROVENANCE OF THE CONSTANTS  (replace with a local refit if the target
 * hardware differs substantially -- these were measured on one machine)
 * ---------------------------------------------------------------------------
 * C0_A, C0_B    fitted over 20920 timed synthetic configurations spanning
 *               d in {4,8,16,30,60} and covariance spectra from a single
 *               dominant eigenvalue to perfectly flat. Per-d fitted values
 *               were 4867/4541/3522/2119/881, giving c0 ~ 14808 * d^-0.62.
 *               Applied WITHOUT refitting to 14 real datasets spanning
 *               d = 3..3072, the resulting speedup prediction achieved a
 *               median relative error of 11.3% and rank correlation +0.924.
 *
 * BAND_A,BAND_B fitted on the same real datasets; median error of the band
 *               estimate itself is ~14%, with individual misses up to 44%.
 *               Adequate for the decision because the SNN/brute boundary sits
 *               near band = 1, far from where the estimate is weakest.
 *
 * KD_MAX_DIM    k-d tree measured faster than SNN only at d = 3, and only on
 *               one of the two datasets available there (road3d, geometric
 *               mean 1.056 in the tree's favour across both). Above d = 10 the
 *               tree is catastrophic for radius search -- 0.03x on a d = 27
 *               dataset -- so this bound is deliberately conservative.
 *
 * ---------------------------------------------------------------------------
 * SCOPE
 * ---------------------------------------------------------------------------
 * Radius search only. k-nearest-neighbour selection is unchanged: SNN as
 * implemented here answers fixed-radius queries, and the cost model above is
 * calibrated against radius-search timings.
 *
 * A caveat worth recording: on the 14-dataset suite SNN was fastest on 81% of
 * (dataset, radius) pairs, and always choosing SNN scored within 1% of the
 * full rule below. The value of this heuristic is in bounding the worst case,
 * not in raising the average.
 */

#include "aoclda.h"
#include "da_std.hpp"
#include "macros.h"
#include "nearest_neighbors_types.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ARCH {

namespace da_neighbors {

namespace heuristic {

/* --- model constants ---------------------------------------------------- */

constexpr double C0_A = 14808.0;  /* c0(d) = C0_A * d^C0_B                   */
constexpr double C0_B = -0.62;
constexpr double BAND_A = 1.547;  /* band ~ erf(BAND_A * ratio^BAND_B)       */
constexpr double BAND_B = 1.107;

constexpr da_int KD_MAX_DIM = 5;       /* k-d tree preferred at or below this */
constexpr da_int KD_BAND_MAX_DIM = 10; /* ... or below this when band is high */
constexpr double KD_BAND_THRESHOLD = 0.8;

/* Distance sampling. The work is m_queries * m_index * d multiply-accumulates,
   so the sample is shrunk as d grows to keep the decision cheap. Floors are
   set so the median remains a usable estimate (>= 8192 pairs). */
constexpr da_int SAMPLE_MAX_QUERIES = 128;
constexpr da_int SAMPLE_MAX_INDEX = 512;
constexpr da_int SAMPLE_MIN_QUERIES = 64;
constexpr da_int SAMPLE_MIN_INDEX = 128;
constexpr da_int SAMPLE_WORK_BUDGET = 25000000; /* multiply-accumulates */

/* --- is this metric one SNN can answer exactly? ------------------------- */

template <typename T>
inline bool metric_supports_snn(da_int metric, T p) {
    /* The projection bound is |<v,q> - <v,x>| <= ||q - x||, an L2 statement,
       so only Euclidean-family metrics give exact results. */
    return metric == da_euclidean || metric == da_euclidean_gemm ||
           metric == da_sqeuclidean || metric == da_sqeuclidean_gemm ||
           (metric == da_minkowski && p == T(2.0));
}

template <typename T>
inline bool metric_supports_tree(da_int metric, T p) {
    /* Mirrors the incompatibility check in set_params(). */
    if (metric == da_cosine || metric == da_sqeuclidean ||
        metric == da_sqeuclidean_gemm)
        return false;
    if (metric == da_minkowski && p < T(1.0))
        return false;
    return true;
}

/* --- cost model --------------------------------------------------------- */

inline double snn_overhead_candidates(da_int n_features) {
    return C0_A * std::pow(static_cast<double>(std::max<da_int>(n_features, 1)),
                           C0_B);
}

/* Predicted SNN speedup over brute force. Above 1.0 means SNN should win. */
inline double predicted_snn_speedup(double band, da_int n_samples,
                                    da_int n_features) {
    const double n = static_cast<double>(n_samples);
    const double denom = band * n + snn_overhead_candidates(n_features);
    return (denom > 0.0) ? n / denom : 1.0;
}

/* --- band estimate ------------------------------------------------------ */

/* Median Euclidean distance between sampled query and index points.
   Column-major: element (i,j) of X is X[i + j*ldx].

   Deterministic stride sampling rather than a PRNG: the decision must be
   reproducible across runs, and the caller has no seed to give us. */
template <typename T>
T sampled_median_distance(da_int n_samples, da_int n_features, const T *X_train,
                          da_int ldx_train, da_int n_queries, const T *X_test,
                          da_int ldx_test) {
    if (n_samples < 1 || n_queries < 1 || n_features < 1)
        return T(0);

    /* Shrink the sample as d grows, subject to floors. */
    da_int budget_pairs =
        std::max<da_int>(SAMPLE_WORK_BUDGET / std::max<da_int>(n_features, 1),
                         SAMPLE_MIN_QUERIES * SAMPLE_MIN_INDEX);
    da_int m_q = std::min<da_int>(SAMPLE_MAX_QUERIES, n_queries);
    da_int m_i = std::min<da_int>(SAMPLE_MAX_INDEX, n_samples);
    if (static_cast<long long>(m_q) * m_i > budget_pairs) {
        /* keep the query:index ratio roughly 1:4 while hitting the budget */
        double scale = std::sqrt(static_cast<double>(budget_pairs) /
                                 (static_cast<double>(m_q) * m_i));
        m_q = std::max<da_int>(SAMPLE_MIN_QUERIES,
                               static_cast<da_int>(m_q * scale));
        m_i = std::max<da_int>(SAMPLE_MIN_INDEX,
                               static_cast<da_int>(m_i * scale));
        m_q = std::min<da_int>(m_q, n_queries);
        m_i = std::min<da_int>(m_i, n_samples);
    }

    const da_int q_stride = std::max<da_int>(1, n_queries / m_q);
    const da_int i_stride = std::max<da_int>(1, n_samples / m_i);

    std::vector<T> d2;
    try {
        d2.reserve(static_cast<size_t>(m_q) * m_i);
    } catch (std::bad_alloc const &) {
        return T(0); /* caller treats 0 as "no estimate available" */
    }

    for (da_int qi = 0, qc = 0; qi < n_queries && qc < m_q;
         qi += q_stride, ++qc) {
        for (da_int ii = 0, ic = 0; ii < n_samples && ic < m_i;
             ii += i_stride, ++ic) {
            T acc = T(0);
            for (da_int j = 0; j < n_features; j++) {
                const T diff = X_test[qi + j * ldx_test] -
                               X_train[ii + j * ldx_train];
                acc += diff * diff;
            }
            d2.push_back(acc);
        }
    }
    if (d2.empty())
        return T(0);

    const size_t mid = d2.size() / 2;
    std::nth_element(d2.begin(), d2.begin() + mid, d2.end());
    return std::sqrt(std::max(d2[mid], T(0)));
}

/* Band fraction estimate in [0,1], or a negative value if unavailable. */
template <typename T>
double estimate_band(da_int n_samples, da_int n_features, const T *X_train,
                     da_int ldx_train, da_int n_queries, const T *X_test,
                     da_int ldx_test, T radius) {
    if (radius <= T(0))
        return 0.0; /* zero radius prunes everything the projection can */

    const T typical = sampled_median_distance(n_samples, n_features, X_train,
                                              ldx_train, n_queries, X_test,
                                              ldx_test);
    if (!(typical > T(0)))
        return -1.0; /* degenerate or allocation failure: no estimate */

    const double ratio = static_cast<double>(radius) / static_cast<double>(typical);
    const double band = std::erf(BAND_A * std::pow(ratio, BAND_B));
    return std::min(1.0, std::max(0.0, band));
}

/* --- the rule ----------------------------------------------------------- */

/* Radius-free part. Everything decidable from geometry and metric alone;
   this is what set_neighbors_algorithm() can use, since the query radius is
   not known when set_params() runs. */
template <typename T>
inline da_int select_radius_algorithm_static(da_int n_samples, da_int n_features,
                                             da_int metric, T p) {
    (void)n_samples;
    if (!metric_supports_snn(metric, p))
        return da_neighbors_types::nn_algorithm::brute;
    if (n_features <= KD_MAX_DIM && metric_supports_tree(metric, p))
        return da_neighbors_types::nn_algorithm::kd_tree;
    return da_neighbors_types::nn_algorithm::snn;
}

/* Radius-aware refinement. Call once the query radius is known; returns the
   algorithm to use, having sampled a small block of distances to estimate the
   band fraction. Falls back to the static rule if the estimate is unavailable. */
template <typename T>
da_int select_radius_algorithm(da_int n_samples, da_int n_features,
                               const T *X_train, da_int ldx_train,
                               da_int n_queries, const T *X_test,
                               da_int ldx_test, T radius, da_int metric, T p) {
    if (!metric_supports_snn(metric, p))
        return da_neighbors_types::nn_algorithm::brute;

    const bool tree_ok = metric_supports_tree(metric, p);
    if (n_features <= KD_MAX_DIM && tree_ok)
        return da_neighbors_types::nn_algorithm::kd_tree;

    const double band = estimate_band(n_samples, n_features, X_train, ldx_train,
                                      n_queries, X_test, ldx_test, radius);
    if (band < 0.0) /* no estimate; default to SNN as the safer choice */
        return da_neighbors_types::nn_algorithm::snn;

    if (n_features <= KD_BAND_MAX_DIM && band > KD_BAND_THRESHOLD && tree_ok)
        return da_neighbors_types::nn_algorithm::kd_tree;

    return (predicted_snn_speedup(band, n_samples, n_features) > 1.0)
               ? da_neighbors_types::nn_algorithm::snn
               : da_neighbors_types::nn_algorithm::brute;
}

} // namespace heuristic

} // namespace da_neighbors

} // namespace ARCH

#endif // NEAREST_NEIGHBORS_HEURISTIC_HPP
