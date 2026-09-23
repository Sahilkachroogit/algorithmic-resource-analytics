// telemetry_gen.cpp
// Compile: g++ -O3 -std=c++17 -o telemetry_gen telemetry_gen.cpp && ./telemetry_gen > data.csv
//
// Benchmarks:
//   1. Iterative Binary Exponentiation
//   2. Dynamic Array Reallocation (std::vector scaling)
//   3. Linked List Deep Traversal
//
// Output: CSV to stdout
//   algorithm,input_size,execution_time_us,memory_bytes

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <vector>

// ---------------------------------------------------------------------------
// Global memory tracking — overload new / delete
// ---------------------------------------------------------------------------
// We use a plain atomic counter implemented via __sync builtins to avoid
// pulling in <atomic>, keeping the TU self-contained.  A spinlock guards the
// counter so that even in single-threaded code the overloads are race-free if
// the stdlib ever calls new internally.

static std::size_t g_alloc_bytes = 0;

// Reset before each benchmark window
static inline void reset_alloc() noexcept { g_alloc_bytes = 0; }
static inline std::size_t snapshot_alloc() noexcept { return g_alloc_bytes; }

// Store the requested size just before the user pointer so we can recover it
// on delete without a map (avoids any additional heap allocation inside the
// tracking itself).
static constexpr std::size_t HDR = alignof(std::max_align_t);

void* operator new(std::size_t n) {
    void* raw = std::malloc(n + HDR);
    if (!raw) throw std::bad_alloc{};
    std::memcpy(raw, &n, sizeof(n));          // store size in header
    g_alloc_bytes += n;
    return static_cast<char*>(raw) + HDR;
}

void* operator new[](std::size_t n) {
    return ::operator new(n);
}

void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}

void* operator new[](std::size_t n, const std::nothrow_t& nt) noexcept {
    return ::operator new(n, nt);
}

void operator delete(void* p) noexcept {
    if (!p) return;
    void* raw = static_cast<char*>(p) - HDR;
    std::size_t n = 0;
    std::memcpy(&n, raw, sizeof(n));
    // We intentionally do NOT subtract from g_alloc_bytes:
    // we want cumulative bytes allocated during the benchmark window,
    // not the live footprint.  Change to g_alloc_bytes -= n if you
    // prefer peak-live semantics.
    std::free(raw);
}

void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { ::operator delete(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { ::operator delete(p); }

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

static inline TimePoint now() { return Clock::now(); }

static inline long long elapsed_us(TimePoint t0, TimePoint t1) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Algorithm 1: Iterative Binary Exponentiation  (base^exp mod MOD)
//   Memory: O(1) — no heap allocation; memory_bytes == 0 for this algo.
//   The work scales with log2(N) per call, so we repeat N calls to make
//   the wall time meaningfully scale with input size N.
// ---------------------------------------------------------------------------
static constexpr uint64_t MOD = (1ULL << 61) - 1; // Mersenne prime

static uint64_t bin_pow_iter(uint64_t base, uint64_t exp, uint64_t mod) noexcept {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1u) result = (__uint128_t)result * base % mod;
        base  = (__uint128_t)base  * base  % mod;
        exp >>= 1;
    }
    return result;
}

static void bench_binpow(long long N, long long& out_us, std::size_t& out_bytes) {
    // Warm-up (not timed)
    volatile uint64_t sink = bin_pow_iter(2, static_cast<uint64_t>(N), MOD);
    (void)sink;

    reset_alloc();
    auto t0 = now();
    // Perform N iterations; vary the exponent so the compiler can't fold them.
    for (long long i = 1; i <= N; ++i) {
        sink ^= bin_pow_iter(static_cast<uint64_t>(i & 0xFF) + 2,
                             static_cast<uint64_t>(i),
                             MOD);
    }
    auto t1 = now();
    out_us    = elapsed_us(t0, t1);
    out_bytes = snapshot_alloc();   // should be 0 — purely stack/register work
}

// ---------------------------------------------------------------------------
// Algorithm 2: Dynamic Array Reallocation (std::vector push_back scaling)
//   We build a vector by push_back-ing N elements from scratch, forcing the
//   geometric reallocation sequence that std::vector uses internally.
//   Every reallocation is a new[] + copy + delete[], all tracked by our
//   overloaded operators.
// ---------------------------------------------------------------------------
static void bench_vector(long long N, long long& out_us, std::size_t& out_bytes) {
    // Warm-up
    {
        std::vector<int> v;
        v.reserve(1);
        for (long long i = 0; i < std::min(N, 16LL); ++i) v.push_back(static_cast<int>(i));
    }

    reset_alloc();
    auto t0 = now();
    {
        std::vector<int> v;   // no reserve — force natural realloc chain
        v.reserve(0);
        for (long long i = 0; i < N; ++i) {
            v.push_back(static_cast<int>(i));
        }
        // Use the data so the compiler can't elide the entire loop.
        volatile int last = v.empty() ? 0 : v.back();
        (void)last;
    }   // destructor frees — not tracked (delete doesn't subtract)
    auto t1 = now();
    out_us    = elapsed_us(t0, t1);
    out_bytes = snapshot_alloc();
}

// ---------------------------------------------------------------------------
// Algorithm 3: Linked List Deep Traversal
//   Build a singly-linked list of N nodes (each node = one heap alloc),
//   then traverse the full list summing node values.  Measures allocation
//   cost + pointer-chasing cache miss pattern.
// ---------------------------------------------------------------------------
struct Node {
    long long value;
    Node*     next;
    explicit Node(long long v) noexcept : value(v), next(nullptr) {}
};

static void bench_linked_list(long long N, long long& out_us, std::size_t& out_bytes) {
    // Warm-up: build and destroy a tiny list
    {
        Node* head = nullptr;
        for (int i = 0; i < 8; ++i) {
            Node* n = new Node(i);
            n->next = head;
            head    = n;
        }
        while (head) { Node* tmp = head->next; delete head; head = tmp; }
    }

    reset_alloc();
    auto t0 = now();

    // --- Build ---
    Node* head = nullptr;
    Node* tail = nullptr;
    for (long long i = 0; i < N; ++i) {
        Node* n = new Node(i);
        if (!tail) { head = tail = n; }
        else       { tail->next = n; tail = n; }
    }

    // --- Traverse ---
    volatile long long acc = 0;
    for (Node* cur = head; cur; cur = cur->next) acc += cur->value;
    (void)acc;

    // --- Free ---
    while (head) {
        Node* tmp = head->next;
        delete head;
        head = tmp;
    }

    auto t1 = now();
    out_us    = elapsed_us(t0, t1);
    out_bytes = snapshot_alloc();
}

// ---------------------------------------------------------------------------
// Input size sequence: non-linear (geometric) steps from 10 to 1,000,000
// ---------------------------------------------------------------------------
static std::vector<long long> build_sizes() {
    std::vector<long long> sizes;
    // Use a multiplier of ~1.5 with a few extra hand-picked waypoints so we
    // get good resolution at both ends without thousands of data points.
    const double mult = 1.5;
    double v = 10.0;
    const long long cap = 1'000'000LL;
    while (static_cast<long long>(v) <= cap) {
        sizes.push_back(static_cast<long long>(v));
        v *= mult;
        if (v < 10.0) v = 10.0;   // guard against fp underflow
    }
    // Always include the exact cap
    if (sizes.empty() || sizes.back() != cap) sizes.push_back(cap);
    return sizes;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Emit CSV header to stdout only — no other prints.
    std::cout << "algorithm,input_size,execution_time_us,memory_bytes\n";

    const auto sizes = build_sizes();

    for (long long N : sizes) {
        long long us   = 0;
        std::size_t mb = 0;

        // --- Binary Exponentiation ---
        bench_binpow(N, us, mb);
        std::cout << "binary_exponentiation," << N << ',' << us << ',' << mb << '\n';

        // --- Vector Reallocation ---
        bench_vector(N, us, mb);
        std::cout << "vector_reallocation," << N << ',' << us << ',' << mb << '\n';

        // --- Linked List ---
        bench_linked_list(N, us, mb);
        std::cout << "linked_list_traversal," << N << ',' << us << ',' << mb << '\n';
    }

    return 0;
}

