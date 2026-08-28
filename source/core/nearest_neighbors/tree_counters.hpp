/* Research instrumentation for tree traversal. Compiled out entirely unless
 * DA_TREE_COUNTERS is defined, so the timing build is unaffected.
 *
 * Counts are deterministic (they do not depend on thread scheduling), so a
 * single run per configuration is sufficient - no repeats, and no exposure to
 * the machine's timing noise.
 *
 * Per-thread counters are incremented in the hot path with no synchronisation.
 * They are folded into the totals once per thread, after the omp for barrier.
 */
#ifndef DA_TREE_COUNTERS_HPP
#define DA_TREE_COUNTERS_HPP

#ifdef DA_TREE_COUNTERS

#include <cstdint>

namespace da_tree_counters {

/* Per-thread. Function-local statics keep this header-only and C++11-safe. */
inline uint64_t &node_visits() { static thread_local uint64_t v = 0; return v; }
inline uint64_t &dist_evals() { static thread_local uint64_t v = 0; return v; }
inline uint64_t &dims_scanned() { static thread_local uint64_t v = 0; return v; }

/* Shared totals. Written once per thread under a critical section. */
inline uint64_t &total_node_visits() { static uint64_t v = 0; return v; }
inline uint64_t &total_dist_evals() { static uint64_t v = 0; return v; }
inline uint64_t &total_dims_scanned() { static uint64_t v = 0; return v; }

} // namespace da_tree_counters

#define DA_COUNT_NODE() (++da_tree_counters::node_visits())
#define DA_COUNT_DIST() (++da_tree_counters::dist_evals())
#define DA_COUNT_DIMS(n) (da_tree_counters::dims_scanned() += (uint64_t)(n))

#else

#define DA_COUNT_NODE() ((void)0)
#define DA_COUNT_DIST() ((void)0)
#define DA_COUNT_DIMS(n) ((void)0)

#endif // DA_TREE_COUNTERS
#endif // DA_TREE_COUNTERS_HPP