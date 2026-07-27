/*
 * Sorting-based nearest neighbours (SNN) index for exact fixed-radius search.
 *
 * Reference: X. Chen & S. Guettel, "Fast and exact fixed-radius neighbor
 * search based on sorting", PeerJ Computer Science 10:e1929 (2024).
 *
 * Index (build):
 *   1. column means mu of the training data A
 *   2. first right singular vector v1 of the centred data (A - 1*mu^T),
 *      via the library's randomized SVD (da_random_svd, k = 1)
 *   3. scores s = A * v1 (uncentred projections: the mean shift cancels in
 *      score differences, which is all the pruning bound uses)
 *   4. argsort(s) -> perm; scores stored sorted ascending
 *   5. X_sorted: rows of A gathered in sorted order (original, uncentred
 *      coordinates) so any projection interval [lo, hi) is a contiguous
 *      column-major submatrix suitable for a single BLAS-3 (gemm) distance
 *      kernel call.
 *
 * Query (radius_neighbors):
 *   For each query q, every point x with ||q - x|| <= eps satisfies
 *   |<v1, q> - <v1, x>| <= ||q - x|| <= eps   (Cauchy-Schwarz, ||v1|| = 1),
 *   so all true neighbours lie in the projection band
 *   [s_q - eps, s_q + eps]. Queries are sorted by projection, processed in
 *   blocks; each block computes one pairwise-distance kernel call over the
 *   union band of its queries, then each query filters its own sub-band
 *   against the (squared) radius.
 */


#include "aoclda.h"
#include "da_error.hpp"
#include "da_vector.hpp"
#include "macros.h"
#include <vector>

namespace ARCH {

namespace da_snn {

template <typename T> class snn_index {
  public:
    /* Light constructor: stores geometry and pointers only; the heavy work
       (SVD, sort, gather) happens in build() so allocation and numerical
       failures are reported through da_status rather than exceptions. */
    snn_index(da_int n_samples, da_int n_features, const T *A, da_int lda,
              da_metric internal_metric, T minkowski_p,
              da_errors::da_error_t *err)
        : n_samples(n_samples), n_features(n_features), lda(lda), A(A),
          internal_metric(internal_metric), p(minkowski_p), err(err) {}

    /* Build the index: mu, v1 (da_random_svd), scores, perm, X_sorted. */
    da_status build();

    /* Exact fixed-radius query.
       band_eps        : plain Euclidean radius (projection band half-width)
       working_radius  : threshold the kernel distances are compared against
                         (equals band_eps^2 when the kernel metric is a
                         squared-Euclidean variant; the caller - the neighbors
                         class - derives both from the user radius and its
                         get_squares flag)
       rnn_indices / rnn_distances : caller-resized to n_queries; results are
                         appended per query, matching the brute-force kernel's
                         contract (distances stored exactly as the kernel
                         produced them; any final sqrt is applied downstream
                         by the existing extraction code). */
    da_status radius_neighbors(da_int n_queries, da_int n_features_in,
                               const T *X_test, da_int ldx_test, T band_eps,
                               T working_radius,
                               std::vector<da_vector::da_vector<da_int>> &rnn_indices,
                               std::vector<da_vector::da_vector<T>> &rnn_distances,
                               bool return_distances,
                               da_errors::da_error_t *err);

  private:
    da_int n_samples = 0;
    da_int n_features = 0;
    da_int lda = 0;
    const T *A = nullptr; /* training data (borrowed; used during build only) */
    da_metric internal_metric;
    T p; /* Minkowski power, forwarded to the distance kernel */
    da_errors::da_error_t *err = nullptr;

    /* Index state (persists between build and queries) */
    std::vector<T> pc;          /* v1, length n_features                     */
    std::vector<T> scores;      /* sorted projections, length n_samples      */
    std::vector<da_int> perm;   /* sorted position -> original row index     */
    std::vector<T> X_sorted;    /* n_samples x n_features, column-major,
                                   rows in sorted-projection order           */
};

} // namespace da_snn

} // namespace ARCH
