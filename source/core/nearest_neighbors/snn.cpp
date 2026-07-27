/*
 * Implementation of the sorting-based nearest neighbours (SNN) index.
 * See snn.hpp for the algorithm description and reference.
 */

#include "snn.hpp"
#include "da_cblas.hh"
#include "da_error.hpp"
#include "da_omp.hpp"
#include "da_utils.hpp"
#include "macros.h"
#include "pairwise_distances.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

/* Query block size for the OpenMP loop; matches the brute-force kernel's
   X_test blocking so the two paths have comparable threading granularity. */
#define SNN_QUERY_BLOCK_SIZE da_int(256)

namespace ARCH {

namespace da_snn {

template <typename T> da_status snn_index<T>::build() {

    if (n_samples < 1 || n_features < 1)
        return da_error_bypass(this->err, da_status_invalid_input,
                               "SNN index requires at least one sample and one "
                               "feature.");

    /* ------------------------------------------------------------------
     * 1. Column means, and the centred copy W = A - 1*mu^T (transient).
     *    W is required only for the SVD: the principal direction must be
     *    computed on centred data, but the projections and the distance
     *    computations both run on the original coordinates.
     * ------------------------------------------------------------------ */
    std::vector<T> mu, W, scores_raw, u, v_prev;
    try {
        mu.resize(n_features);
        W.resize(static_cast<size_t>(n_samples) * n_features);
        scores_raw.resize(n_samples);
        u.resize(n_samples);
        v_prev.resize(n_features);
        pc.resize(n_features);
        scores.resize(n_samples);
        perm.resize(n_samples);
        X_sorted.resize(static_cast<size_t>(n_samples) * n_features);
    } catch (std::bad_alloc const &) {
        return da_error(this->err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }

    for (da_int j = 0; j < n_features; j++) {
        const T *col = A + static_cast<size_t>(j) * lda;
        T acc = T(0);
        for (da_int i = 0; i < n_samples; i++)
            acc += col[i];
        mu[j] = acc / static_cast<T>(n_samples);
        T *wcol = W.data() + static_cast<size_t>(j) * n_samples;
        for (da_int i = 0; i < n_samples; i++)
            wcol[i] = col[i] - mu[j];
    }

    /* ------------------------------------------------------------------
     * 2. First right singular vector of W by power iteration on W^T W:
     *      u <- W v ;  v <- W^T u ;  v <- v / ||v||
     *    Two BLAS-2 calls per iteration; no dependency on any external
     *    decomposition routine. The start vector is deterministic, so
     *    the index is reproducible.
     *
     *    IMPORTANT: any unit vector gives EXACT query results -- the
     *    Cauchy-Schwarz projection bound holds for every direction. The
     *    leading singular vector merely maximises the spread of the
     *    projections and hence how tightly the bands prune. The tolerance
     *    and iteration cap below therefore trade index-build time against
     *    query speed, never correctness.
     * ------------------------------------------------------------------ */
    {
        /* Deterministic pseudo-random start: a constant vector can be
           exactly orthogonal to the leading singular vector on symmetric
           data, which would stall the iteration. */
        unsigned int state = 12345u;
        for (da_int j = 0; j < n_features; j++) {
            state = 1664525u * state + 1013904223u;
            pc[j] = static_cast<T>((state >> 8) & 0xFFFFu) / T(65535) - T(0.5);
        }
        T nrm = da_blas::cblas_nrm2(n_features, pc.data(), 1);
        if (!(nrm > T(0))) {
            std::fill(pc.begin(), pc.end(), T(0));
            pc[0] = T(1);
        } else {
            for (da_int j = 0; j < n_features; j++)
                pc[j] /= nrm;
        }
    }

    {
        /* A modest budget is deliberate. Accuracy of v1 affects only how
           tightly the bands prune, never correctness, and the objective is
           flat near its maximum -- a direction within ~1e-2 rad of v1 prunes
           essentially as well. Moreover, when the iteration converges slowly
           the spectrum is flat, i.e. the data has no dominant direction and
           NO projection prunes well; spending more iterations there would
           buy nothing. Each iteration costs two passes over the centred
           copy, so the cap also bounds index-build time. */
        const da_int max_iter = 20;
        const T tol = static_cast<T>(1.0e-4);
        for (da_int it = 0; it < max_iter; it++) {
            v_prev = pc;
            /* u = W * v */
            da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, n_samples,
                                n_features, T(1), W.data(), n_samples,
                                pc.data(), 1, T(0), u.data(), 1);
            /* v = W^T * u */
            da_blas::cblas_gemv(CblasColMajor, CblasTrans, n_samples,
                                n_features, T(1), W.data(), n_samples, u.data(),
                                1, T(0), pc.data(), 1);
            T nrm = da_blas::cblas_nrm2(n_features, pc.data(), 1);
            if (!(nrm > std::numeric_limits<T>::min())) {
                /* Degenerate data (e.g. every point identical): any unit
                   vector is admissible. All projections then coincide, the
                   band spans the index and SNN reduces to brute force --
                   slower, but still exact. */
                std::fill(pc.begin(), pc.end(), T(0));
                pc[0] = T(1);
                break;
            }
            for (da_int j = 0; j < n_features; j++)
                pc[j] /= nrm;
            /* Converged when the direction stops moving (sign-insensitive). */
            T dot = T(0);
            for (da_int j = 0; j < n_features; j++)
                dot += pc[j] * v_prev[j];
            if (T(1) - std::abs(dot) < tol)
                break;
        }
    }

    W = std::vector<T>{};      /* release the centred copy */
    u = std::vector<T>{};
    v_prev = std::vector<T>{};

    /* ------------------------------------------------------------------
     * 3. Projections of the ORIGINAL data: s = A * v1 (one gemv).
     * ------------------------------------------------------------------ */
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, n_samples, n_features,
                        T(1), A, lda, pc.data(), 1, T(0), scores_raw.data(), 1);

    /* ------------------------------------------------------------------
     * 4. Sort: perm = argsort(scores_raw); scores = sorted projections.
     * ------------------------------------------------------------------ */
    std::iota(perm.begin(), perm.end(), da_int(0));
    std::sort(perm.begin(), perm.end(), [&scores_raw](da_int a, da_int b) {
        return scores_raw[a] < scores_raw[b];
    });
    for (da_int i = 0; i < n_samples; i++)
        scores[i] = scores_raw[perm[i]];

    /* ------------------------------------------------------------------
     * 5. Gather rows into sorted order (column-wise for cache-friendly
     *    writes). X_sorted holds ORIGINAL coordinates: distances are
     *    computed on real data; centring was for the SVD only.
     * ------------------------------------------------------------------ */
    for (da_int j = 0; j < n_features; j++) {
        const T *src = A + static_cast<size_t>(j) * lda;
        T *dst = X_sorted.data() + static_cast<size_t>(j) * n_samples;
        for (da_int i = 0; i < n_samples; i++)
            dst[i] = src[perm[i]];
    }

    return da_status_success;
}

template <typename T>
da_status snn_index<T>::radius_neighbors(
    da_int n_queries, da_int n_features_in, const T *X_test, da_int ldx_test,
    T band_eps, T working_radius,
    std::vector<da_vector::da_vector<da_int>> &rnn_indices,
    std::vector<da_vector::da_vector<T>> &rnn_distances, bool return_distances,
    da_errors::da_error_t *err) {

    if (n_features_in != this->n_features)
        return da_error_bypass(err, da_status_invalid_input,
                               "The number of features in the query data does "
                               "not match the training data.");
    if (n_queries < 1)
        return da_status_success;

    /* Small relative slack on the band bounds guards against floating-point
       rounding in the projections excluding a true neighbour sitting exactly
       on the radius. Mathematically the band is exact; the slack only ever
       admits a few extra candidates, which the distance filter removes. */
    T slack = std::abs(band_eps) * std::numeric_limits<T>::epsilon() * T(8);
    T band = band_eps + slack;

    /* ------------------------------------------------------------------
     * 1. Project all queries: qs = X_test * v1 (one gemv), then sort the
     *    queries by projection so consecutive queries have overlapping
     *    bands, keeping each block's union band tight.
     * ------------------------------------------------------------------ */
    std::vector<T> qs;
    std::vector<da_int> qorder;
    try {
        qs.resize(n_queries);
        qorder.resize(n_queries);
    } catch (std::bad_alloc const &) {
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    }
    da_blas::cblas_gemv(CblasColMajor, CblasNoTrans, n_queries, n_features,
                        T(1), X_test, ldx_test, pc.data(), 1, T(0), qs.data(),
                        1);
    std::iota(qorder.begin(), qorder.end(), da_int(0));
    std::sort(qorder.begin(), qorder.end(),
              [&qs](da_int a, da_int b) { return qs[a] < qs[b]; });

    /* ------------------------------------------------------------------
     * 2. Block the sorted queries; one thread owns one block, so each
     *    query is written by exactly one thread and no result merging is
     *    needed (unlike the brute kernel, whose training-side blocking
     *    splits a query's neighbours across threads).
     * ------------------------------------------------------------------ */
    da_int qblock_size = std::min(SNN_QUERY_BLOCK_SIZE, n_queries);
    da_int qblock_rem, n_qblocks;
    ARCH::da_utils::blocking_scheme(n_queries, qblock_size, n_qblocks,
                                    qblock_rem);
    da_int n_threads = ARCH::da_utils::get_n_threads_loop(n_qblocks);

    da_int threading_error = 0;

    /* Locals for the shared clause (avoids member variables in OpenMP
       data-sharing clauses). */
    const T *scores_ptr = scores.data();
    const T *Xs_ptr = X_sorted.data();
    const da_int *perm_ptr = perm.data();
    da_int ns = n_samples, nf = n_features;
    T minkowski_p = this->p;
    da_metric metric = this->internal_metric;

#pragma omp parallel num_threads(n_threads) default(none)                        \
    shared(threading_error, rnn_indices, rnn_distances, return_distances,        \
               qblock_size, qblock_rem, n_qblocks, n_queries, qs, qorder,        \
               scores_ptr, Xs_ptr, perm_ptr, ns, nf, minkowski_p, metric,        \
               band, working_radius, X_test, ldx_test)
    {
        da_int local_error = 0;
        std::vector<T> Qbuf, D;
        try {
            Qbuf.resize(static_cast<size_t>(qblock_size) * nf);
        } catch (std::bad_alloc const &) {
#pragma omp atomic write
            threading_error = 1;
        }

#pragma omp for schedule(dynamic)
        for (da_int b = 0; b < n_qblocks; b++) {
#pragma omp atomic read
            local_error = threading_error;
            if (local_error != 0)
                continue;

            da_int local_block = qblock_size;
            if (b == n_qblocks - 1 && qblock_rem > 0)
                local_block = qblock_rem;
            da_int qstart = b * qblock_size;

            /* Union band of this block: the queries are sorted by
               projection, so the block minimum and maximum are its first
               and last members. */
            T lo_val = qs[qorder[qstart]] - band;
            T hi_val = qs[qorder[qstart + local_block - 1]] + band;
            da_int lo = static_cast<da_int>(
                std::lower_bound(scores_ptr, scores_ptr + ns, lo_val) -
                scores_ptr);
            da_int hi = static_cast<da_int>(
                std::upper_bound(scores_ptr, scores_ptr + ns, hi_val) -
                scores_ptr);
            if (lo >= hi)
                continue; /* no candidates for any query in this block */
            da_int band_size = hi - lo;

            /* Gather the block's query rows (scattered by the sort) into a
               contiguous column-major buffer with ld = local_block. */
            for (da_int j = 0; j < nf; j++) {
                T *dst = Qbuf.data() + static_cast<size_t>(j) * local_block;
                const T *src = X_test + static_cast<size_t>(j) * ldx_test;
                for (da_int qq = 0; qq < local_block; qq++)
                    dst[qq] = src[qorder[qstart + qq]];
            }

            /* One BLAS-3 distance kernel call for the whole block over the
               union band: D is (band_size x local_block), column-major,
               D[ii + qq*band_size] = dist(X_sorted[lo+ii], query qq). Same
               kernel and metric the brute path uses -> same substrate. */
            try {
                D.resize(static_cast<size_t>(band_size) * local_block);
            } catch (std::bad_alloc const &) {
#pragma omp atomic write
                threading_error = 1;
                continue;
            }
            da_status thd_status =
                ARCH::da_metrics::pairwise_distances::pairwise_distance_kernel(
                    da_order::column_major, band_size, local_block, nf,
                    Xs_ptr + lo, ns, Qbuf.data(), local_block, D.data(),
                    band_size, minkowski_p, metric);
            if (thd_status != da_status_success) {
#pragma omp atomic write
                threading_error = 1;
                continue;
            }

            /* Per query: restrict to its own sub-band inside [lo, hi) -- by
               the projection bound nothing outside it can be a neighbour --
               then filter against the radius and translate sorted positions
               back to original row indices. */
            for (da_int qq = 0; qq < local_block; qq++) {
                da_int orig = qorder[qstart + qq];
                T q_lo = qs[orig] - band;
                T q_hi = qs[orig] + band;
                da_int l_i = static_cast<da_int>(
                    std::lower_bound(scores_ptr + lo, scores_ptr + hi, q_lo) -
                    scores_ptr);
                da_int h_i = static_cast<da_int>(
                    std::upper_bound(scores_ptr + lo, scores_ptr + hi, q_hi) -
                    scores_ptr);
                const T *Dcol = D.data() + static_cast<size_t>(qq) * band_size;
                for (da_int ii = l_i - lo; ii < h_i - lo; ii++) {
                    if (Dcol[ii] <= working_radius) {
                        try {
                            rnn_indices[orig].push_back(perm_ptr[lo + ii]);
                            if (return_distances)
                                rnn_distances[orig].push_back(Dcol[ii]);
                        } catch (std::bad_alloc const &) {
#pragma omp atomic write
                            threading_error = 1;
                        }
                    }
                }
            }
        } /* end of query blocks */
    }     /* end of parallel region */

    if (threading_error != 0)
        return da_error(err, da_status_memory_error, // LCOV_EXCL_LINE
                        "Memory allocation failed.");
    return da_status_success;
}

template class snn_index<double>;
template class snn_index<float>;

} // namespace da_snn

} // namespace ARCH
