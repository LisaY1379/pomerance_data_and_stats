/*
 * pomerance.c — Find Pomerance triples (p, A, x0) for a given prime p
 *
 * A Pomerance triple (p, A, x0) is defined as follows: p is an odd prime,
 * A and x0 are nonneg integers < p with A ≠ ±2 mod p, such that doubling
 * the projective point (x0:1) on the Montgomery curve By^2 = x^3+Ax^2+x
 * exactly k times yields Z ≡ 0 mod p, where k is the least integer with
 * 2^k > floor(sqrt(p)) + 1 + 2*floor(sqrt(floor(sqrt(p)))).
 *
 * Algorithm: 2-Sylow projection.  For each candidate group order N = 2^k·m
 * in the Hasse interval, multiply a random point by the odd part m first
 * (projecting into the 2-Sylow subgroup), then double.  This gives
 * O(1/sqrt(p)) success probability per random (A, x0) trial.
 *
 * Automatically uses u64 arithmetic for p < 2^63, u128 for p < 2^127.
 *
 * Compile:
 * gcc -O3 -fopenmp -o pomerance pomerance.c -lm
 * gcc -O3 -o pomerance pomerance.c -lm           (single-threaded)
 *
 * Usage:
 * ./pomerance <input_primes.txt> <output_pure.csv> <output_metrics.csv> <num_triples>
 *
 * Reference: https://github.com/AndrewVSutherland/DANGER3
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef uint64_t u64;
typedef __uint128_t u128;

/* ================================================================
 * Parsing / printing u128
 * ================================================================ */

static u128 parse128(const char *s) {
    u128 v = 0;
    while (*s >= '0' && *s <= '9') { v = v * 10 + (*s - '0'); s++; }
    return v;
}

static void sprint128(char *buf, u128 v) {
    if (v == 0) { buf[0] = '0'; buf[1] = '\0'; return; }
    char tmp[50]; int i = 49; tmp[i] = '\0';
    while (v > 0) { tmp[--i] = '0' + (int)(v % 10); v /= 10; }
    strcpy(buf, tmp + i);
}

static void print128(u128 v) { char b[50]; sprint128(b, v); fputs(b, stdout); }

static int digits128(u128 v) { char b[50]; sprint128(b, v); return (int)strlen(b); }

/* ================================================================
 * PRNG (xorshift128+)
 * ================================================================ */

typedef struct { u64 s0, s1; } Rng;

static inline u64 rng64(Rng *r) {
    u64 s1 = r->s0, s0 = r->s1; r->s0 = s0;
    s1 ^= s1 << 23; r->s1 = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
    return r->s1 + s0;
}

/* ================================================================
 * Timer
 * ================================================================ */

static double now_sec(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* ================================================================
 * u64 code path (p < 2^63)
 * ================================================================ */

static inline u64 addmod64(u64 a, u64 b, u64 p) { return a >= p-b ? a-(p-b) : a+b; }
static inline u64 submod64(u64 a, u64 b, u64 p) { return a >= b ? a-b : p-b+a; }
static inline u64 mulmod64(u64 a, u64 b, u64 p) { return (u64)((u128)a*b%(u128)p); }

typedef struct { u64 p, ni, R2, one; } Mont64;

static void m64_init(Mont64 *m, u64 p) {
    m->p = p; u64 x = 1;
    for (int i = 0; i < 6; i++) x *= 2 - p * x;
    m->ni = (u64)(0ULL - x);
    m->one = (u64)(((u128)1 << 64) % (u128)p);
    m->R2  = (u64)(((u128)m->one * m->one) % (u128)p);
}

static inline u64 mred64(u128 t, const Mont64 *m) {
    u64 q = (u64)t * m->ni;
    u64 r = (u64)((t + (u128)q * m->p) >> 64);
    return r >= m->p ? r - m->p : r;
}
static inline u64 mm64(u64 a, u64 b, const Mont64 *m) { return mred64((u128)a*b, m); }
static inline u64 toM64(u64 a, const Mont64 *m) { return mm64(a % m->p, m->R2, m); }
static inline u64 frM64(u64 a, const Mont64 *m) { return mred64((u128)a, m); }

static inline void xDBL64(u64 *Xo, u64 *Zo, u64 X, u64 Z, u64 a24, const Mont64 *m) {
    u64 p = m->p;
    u64 u = addmod64(X,Z,p), v = submod64(X,Z,p);
    u = mm64(u,u,m); v = mm64(v,v,m);
    *Xo = mm64(u,v,m);
    u64 w = submod64(u,v,p);
    *Zo = mm64(w, addmod64(v, mm64(a24,w,m), p), m);
}

static inline void xADD64(u64 *Xo, u64 *Zo, u64 X0, u64 Z0, u64 X1, u64 Z1,
                            u64 xP, const Mont64 *m) {
    u64 p = m->p;
    u64 u = mm64(submod64(X0,Z0,p), addmod64(X1,Z1,p), m);
    u64 v = mm64(addmod64(X0,Z0,p), submod64(X1,Z1,p), m);
    u64 s = addmod64(u,v,p), d = submod64(u,v,p);
    *Xo = mm64(s,s,m);
    *Zo = mm64(xP, mm64(d,d,m), m);
}

static void xMUL64(u64 *Xo, u64 *Zo, u64 xP, u64 n, u64 a24, const Mont64 *m) {
    if (n==0) { *Xo=0; *Zo=0; return; }
    if (n==1) { *Xo=xP; *Zo=m->one; return; }
    u64 X0=xP, Z0=m->one, X1, Z1;
    xDBL64(&X1,&Z1,X0,Z0,a24,m);
    int bits = 64 - __builtin_clzll(n);
    for (int i=bits-2; i>=0; i--) {
        if ((n>>i)&1) { xADD64(&X0,&Z0,X0,Z0,X1,Z1,xP,m); xDBL64(&X1,&Z1,X1,Z1,a24,m); }
        else          { xADD64(&X1,&Z1,X0,Z0,X1,Z1,xP,m); xDBL64(&X0,&Z0,X0,Z0,a24,m); }
    }
    *Xo=X0; *Zo=Z0;
}

static int verify64(u64 p, u64 A, u64 x0) {
    u64 q = (u64)sqrtl((long double)p);
    while ((u128)(q+1)*(q+1)<=(u128)p) q++;
    while ((u128)q*q>(u128)p) q--;
    u64 sq = (u64)sqrtl((long double)q);
    while ((sq+1)*(sq+1)<=q) sq++;
    while (sq*sq>q) sq--;
    u64 bound = q+1+2*sq;
    int k=0; u64 v=1; while(v<=bound){k++;v<<=1;}

    if (A%p==2||A%p==p-2) return 0;
    u64 X=x0%p, Z=1;
    for (int i=1; i<=k; i++) {
        u64 X2=mulmod64(X,X,p), Z2=mulmod64(Z,Z,p), XZ=mulmod64(X,Z,p);
        u64 d=submod64(X2,Z2,p), Xn=mulmod64(d,d,p);
        u64 inn=addmod64(addmod64(X2,mulmod64(A,XZ,p),p),Z2,p);
        u64 f4=addmod64(addmod64(XZ,XZ,p),addmod64(XZ,XZ,p),p);
        u64 Zn=mulmod64(f4,inn,p); X=Xn; Z=Zn;
        if (i<k&&Z==0) return 0;
        if (i==k&&Z!=0) return 0;
    }
    return 1;
}

/* ================================================================
 * u128 code path (p < 2^127)
 * ================================================================ */

typedef struct { u128 lo, hi; } u256;

static inline u256 wide_mul(u128 a, u128 b) {
    u64 a0=(u64)a, a1=(u64)(a>>64), b0=(u64)b, b1=(u64)(b>>64);
    u128 ll=(u128)a0*b0, lh=(u128)a0*b1, hl=(u128)a1*b0, hh=(u128)a1*b1;
    u128 mid=lh+hl; u128 carry_mid=(mid<lh)?1:0;
    u128 lo=ll+(mid<<64); u128 carry_lo=(lo<ll)?1:0;
    return (u256){lo, hh+(mid>>64)+(carry_mid<<64)+carry_lo};
}

static inline u256 wide_add(u256 a, u256 b) {
    u128 lo=a.lo+b.lo; return (u256){lo, a.hi+b.hi+((lo<a.lo)?1:0)};
}

typedef struct { u128 p, ni, R2, one; } Mont128;

static void m128_init(Mont128 *m, u128 p) {
    m->p = p;
    u128 x = 1; for (int i = 0; i < 7; i++) x *= 2 - p * x;
    m->ni = (u128)0 - x;
    u128 r = 1;
    for (int i = 0; i < 128; i++) { r<<=1; if(r>=p) r-=p; }
    m->one = r;
    for (int i = 0; i < 128; i++) { r<<=1; if(r>=p) r-=p; }
    m->R2 = r;
}

static inline u128 mred128(u256 T, const Mont128 *m) {
    u128 q = T.lo * m->ni;
    u256 s = wide_add(T, wide_mul(q, m->p));
    u128 t = s.hi;
    return t >= m->p ? t - m->p : t;
}
static inline u128 mm128(u128 a, u128 b, const Mont128 *m) { return mred128(wide_mul(a,b), m); }
static inline u128 toM128(u128 a, const Mont128 *m) { return mm128(a % m->p, m->R2, m); }
static inline u128 frM128(u128 a, const Mont128 *m) { return mred128((u256){a,0}, m); }

static inline u128 addmod128(u128 a, u128 b, u128 p) { u128 s=a+b; return s>=p?s-p:s; }
static inline u128 submod128(u128 a, u128 b, u128 p) { return a>=b?a-b:p-b+a; }

static inline void xDBL128(u128 *Xo, u128 *Zo, u128 X, u128 Z, u128 a24, const Mont128 *m) {
    u128 p=m->p;
    u128 u=addmod128(X,Z,p), v=submod128(X,Z,p);
    u=mm128(u,u,m); v=mm128(v,v,m);
    *Xo=mm128(u,v,m);
    u128 w=submod128(u,v,p);
    *Zo=mm128(w, addmod128(v, mm128(a24,w,m), p), m);
}

static inline void xADD128(u128 *Xo, u128 *Zo, u128 X0, u128 Z0, u128 X1, u128 Z1,
                             u128 xP, const Mont128 *m) {
    u128 p=m->p;
    u128 u=mm128(submod128(X0,Z0,p), addmod128(X1,Z1,p), m);
    u128 v=mm128(addmod128(X0,Z0,p), submod128(X1,Z1,p), m);
    u128 s=addmod128(u,v,p), d=submod128(u,v,p);
    *Xo=mm128(s,s,m);
    *Zo=mm128(xP, mm128(d,d,m), m);
}

static void xMUL128(u128 *Xo, u128 *Zo, u128 xP, u64 n, u128 a24, const Mont128 *m) {
    if (n==0) { *Xo=0; *Zo=0; return; }
    if (n==1) { *Xo=xP; *Zo=m->one; return; }
    u128 X0=xP, Z0=m->one, X1, Z1;
    xDBL128(&X1,&Z1,X0,Z0,a24,m);
    int bits = 64 - __builtin_clzll(n);
    for (int i=bits-2; i>=0; i--) {
        if ((n>>i)&1) { xADD128(&X0,&Z0,X0,Z0,X1,Z1,xP,m); xDBL128(&X1,&Z1,X1,Z1,a24,m); }
        else          { xADD128(&X1,&Z1,X0,Z0,X1,Z1,xP,m); xDBL128(&X0,&Z0,X0,Z0,a24,m); }
    }
    *Xo=X0; *Zo=Z0;
}

static u128 mulmod_slow(u128 a, u128 b, u128 p) {
    u128 r=0; a%=p; b%=p;
    while (b>0) { if(b&1){r+=a;if(r>=p)r-=p;} a+=a;if(a>=p)a-=p; b>>=1; }
    return r;
}

static int verify128(u128 p, u128 A, u128 x0) {
    u64 q = (u64)sqrtl((long double)p);
    while ((u128)(q+1)*(q+1)<=p) q++;
    while ((u128)q*q>p) q--;
    u64 sq = (u64)sqrtl((long double)q);
    while ((sq+1)*(sq+1)<=q) sq++;
    while (sq*sq>q) sq--;
    u64 bound = q+1+2*sq;
    int k=0; u64 v=1; while(v<=bound){k++;v<<=1;}

    if (A%p==2||A%p==p-2) return 0;
    u128 X=x0%p, Z=1;
    for (int i=1; i<=k; i++) {
        u128 X2=mulmod_slow(X,X,p), Z2=mulmod_slow(Z,Z,p), XZ=mulmod_slow(X,Z,p);
        u128 d=submod128(X2,Z2,p), Xn=mulmod_slow(d,d,p);
        u128 inn=addmod128(addmod128(X2,mulmod_slow(A,XZ,p),p),Z2,p);
        u128 f4=addmod128(addmod128(XZ,XZ,p),addmod128(XZ,XZ,p),p);
        u128 Zn=mulmod_slow(f4,inn,p); X=Xn; Z=Zn;
        if (i<k&&Z==0) return 0;
        if (i==k&&Z!=0) return 0;
    }
    return 1;
}

/* ================================================================
 * Shared: compute_k, odd parts, Miller-Rabin (all u128-safe)
 * ================================================================ */

static int compute_k(u128 p) {
    u64 q = (u64)sqrtl((long double)p);
    while ((u128)(q+1)*(q+1)<=p) q++;
    while ((u128)q*q>p) q--;
    u64 sq = (u64)sqrtl((long double)q);
    while ((sq+1)*(sq+1)<=q) sq++;
    while (sq*sq>q) sq--;
    u64 bound = q+1+2*sq;
    int k=0; u64 v=1; while(v<=bound){k++;v<<=1;} return k;
}

static int compute_odd_parts(u128 p, int k, u64 *ms, int max_ms) {
    if (k >= 63) return 0;
    u64 twok = 1ULL << k;
    u128 pp1 = p + 1;
    u64 r = (u64)(pp1 % twok);
    u64 sqrtp = (u64)sqrtl((long double)p);
    while ((u128)(sqrtp+1)*(sqrtp+1)<=p) sqrtp++;
    while ((u128)sqrtp*sqrtp>p) sqrtp--;
    u64 two_sqrtp = sqrtp * 2 + 4;
    u64 residues[2] = { r, (twok - r) % twok };
    int count = 0;
    for (int ri = 0; ri < 2 && count < max_ms; ri++) {
        u64 res = residues[ri];
        for (int sign = -1; sign <= 1; sign += 2) {
            for (u64 j = 0; ; j++) {
                long long tv;
                if (sign > 0) tv = (long long)(res + j * twok);
                else           tv = (long long)res - (long long)((j+1) * twok);
                if (tv > (long long)two_sqrtp || tv < -(long long)two_sqrtp) break;
                //CODE UPDATE: Commented out the line abandoning supersingular elliptic curve
                //if (tv == 0) continue;
                u128 N = pp1;
                if (ri == 0) { if (tv>=0) N-=(u64)tv; else N+=(u64)(-tv); }
                else         { if (tv>=0) N+=(u64)tv; else N-=(u64)(-tv); }
                if (N == 0) continue;
                int v2 = 0; u128 tmp = N;
                while (tmp % 2 == 0) { v2++; tmp /= 2; }
                if (v2 < k) continue;
                if (tmp >> 63) continue;
                u64 odd = (u64)tmp;
                if (odd == 0) continue;
                int dup = 0;
                for (int c = 0; c < count; c++) if (ms[c] == odd) { dup = 1; break; }
                if (!dup) ms[count++] = odd;
            }
        }
    }
    return count;
}

static int is_prime128(u128 n) {
    if (n < 2) return 0; if (n < 4) return 1; if (n % 2 == 0) return 0;
    u128 d = n-1; int r = 0; while (d%2==0) { d/=2; r++; }
    u64 w[] = {2,3,5,7,11,13,17,19,23,29,31,37};
    for (int i = 0; i < 12; i++) {
        u128 a = w[i]; if (a >= n) continue;
        u128 x = 1, b = a;
        for (u128 e = d; e; e >>= 1) {
            if (e & 1) x = mulmod_slow(x,b,n);
            b = mulmod_slow(b,b,n);
        }
        if (x == 1 || x == n-1) continue;
        int ok = 0;
        for (int j = 0; j < r-1; j++) { x = mulmod_slow(x,x,n); if (x==n-1){ok=1;break;} }
        if (!ok) return 0;
    }
    return 1;
}

/* ================================================================
 * Dispatch: search64 / search128
 * ================================================================ */

static int search64(u64 p, int target_count, u64 *out_A, u64 *out_x0, u64 *out_trials) {
    volatile int found_count = 0;

    int k = compute_k(p);
    u64 sqrtp = (u64)sqrtl((long double)p);
    while ((u128)(sqrtp+1)*(sqrtp+1)<=(u128)p) sqrtp++;
    while ((u128)sqrtp*sqrtp>(u128)p) sqrtp--;

    u64 ms[64]; int nms = compute_odd_parts(p, k, ms, 64);
    if (nms == 0) { printf("No valid odd parts.\n"); return found_count; }

    Mont64 mt; m64_init(&mt, p);
    u64 inv4; { u64 r=1,b=4%p; for(u64 e=p-2;e;e>>=1){if(e&1)r=mulmod64(r,b,p);b=mulmod64(b,b,p);} inv4=r; }

    u64 max_trials = (u64)(20.0 * (double)sqrtp / nms);
    if (max_trials < 10000000ULL) max_trials = 10000000ULL;

    u64 thread_trials[256 * 8] = {0};

#pragma omp parallel
    {
        int tid=0, nthr=1;
#ifdef _OPENMP
        tid = omp_get_thread_num(); nthr = omp_get_num_threads();
#endif
        u64 current_time = (u64)time(NULL);
        Rng rng = {
            .s0 = 7364529176530163ULL ^ ((u64)tid * 6364136223846793005ULL) ^ p ^ current_time,
            .s1 = 1442695040888963407ULL ^ ((u64)(tid+1) * 2862933555777941757ULL) ^ (current_time << 32)
        };
        for (int i=0;i<200;i++) rng64(&rng);

        u64 budget = max_trials / nthr + 1;
        u64 A_trials = 0;

        while (found_count < target_count && A_trials < budget) {
            u64 A = rng64(&rng) % p;
            if (A==2||A==p-2) { continue; }

            A_trials++;
            thread_trials[tid * 8] = A_trials;

            u64 a24 = mulmod64(addmod64(A,2,p), inv4, p);
            u64 a24m = toM64(a24, &mt);

            int a_success = 0;

            for (int x_tries = 0; x_tries < 50 && !a_success && found_count < target_count; x_tries++) {
                u64 x0r = rng64(&rng) % p;
                if (x0r < 2) continue;
                u64 x0m = toM64(x0r, &mt);

                for (int mi=0; mi<nms && !a_success && found_count < target_count; mi++) {
                    u64 QX, QZ;
                    xMUL64(&QX, &QZ, x0m, ms[mi], a24m, &mt);
                    if (QZ == 0) continue;
                    u64 CX=QX, CZ=QZ;
                    int zs = -1;
                    for (int s=1; s<=k+10 && s<50; s++) {
                        xDBL64(&CX,&CZ,CX,CZ,a24m,&mt);
                        if (CZ==0) { zs=s; break; }
                    }
                    if (zs < k) continue;

                    int target_z = zs - k;
                    CX=QX; CZ=QZ;
                    for (int s=0; s<target_z; s++) xDBL64(&CX,&CZ,CX,CZ,a24m,&mt);
                    u64 cz = frM64(CZ, &mt);
                    if (cz == 0) continue;
                    u64 czinv; {u64 r2=1,b2=cz;for(u64 e=p-2;e;e>>=1){if(e&1)r2=mulmod64(r2,b2,p);b2=mulmod64(b2,b2,p);}czinv=r2;}
                    u64 xR = mulmod64(frM64(CX,&mt), czinv, p);

                    if (verify64(p, A, xR)) {
#pragma omp critical
                        {
                            int is_dup = 0;
                            for(int j = 0; j < found_count; j++) {
                                if(out_A[j] == A && out_x0[j] == xR) {
                                    is_dup = 1; break;
                                }
                            }
                            if (!is_dup && found_count < target_count) {
                                out_A[found_count] = A;
                                out_x0[found_count] = xR;

                                u64 exact_total_trials = 0;
                                for (int t = 0; t < nthr; t++) {
                                    exact_total_trials += thread_trials[t * 8];
                                }
                                if (exact_total_trials == 0) exact_total_trials = 1;

                                out_trials[found_count] = exact_total_trials;
                                found_count++;
                                a_success = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    return found_count;
}

static int search128(u128 p, int target_count, u128 *out_A, u128 *out_x0, u64 *out_trials) {
    volatile int found_count = 0;

    int k = compute_k(p);
    u64 sqrtp = (u64)sqrtl((long double)p);
    while ((u128)(sqrtp+1)*(sqrtp+1)<=p) sqrtp++;
    while ((u128)sqrtp*sqrtp>p) sqrtp--;

    u64 ms[64]; int nms = compute_odd_parts(p, k, ms, 64);
    if (nms == 0) { printf("No valid odd parts.\n"); return found_count; }

    Mont128 mt; m128_init(&mt, p);
    /* inv4 in Montgomery form */
    u128 inv4_m;
    { u128 four_m=toM128(4,&mt), r=mt.one, b=four_m; u128 e=p-2;
      while(e>0){if(e&1)r=mm128(r,b,&mt);b=mm128(b,b,&mt);e>>=1;} inv4_m=r; }

    u64 max_trials = (u64)(20.0 * (double)sqrtp / nms);
    if (max_trials < 10000000ULL) max_trials = 10000000ULL;

    u64 thread_trials[256 * 8] = {0};

#pragma omp parallel
    {
        int tid=0, nthr=1;
#ifdef _OPENMP
        tid = omp_get_thread_num(); nthr = omp_get_num_threads();
#endif
        u64 current_time = (u64)time(NULL);
        Rng rng = {
            .s0 = 7364529176530163ULL ^ ((u64)tid * 6364136223846793005ULL) ^ (u64)p ^ current_time,
            .s1 = 1442695040888963407ULL ^ ((u64)(tid+1) * 2862933555777941757ULL) ^ (current_time << 32)
        };
        for (int i=0;i<200;i++) rng64(&rng);

        u64 budget = max_trials / nthr + 1;
        u64 A_trials = 0;

        while (found_count < target_count && A_trials < budget) {
            u128 A = (u128)rng64(&rng) | ((u128)rng64(&rng) << 64); A %= p;
            if (A==2 || A==p-2) continue;

            A_trials++;
            thread_trials[tid * 8] = A_trials;

            u128 Ap2_m = toM128(addmod128(A,2,p), &mt);
            u128 a24m = mm128(Ap2_m, inv4_m, &mt);

            int a_success = 0;

            for (int x_tries = 0; x_tries < 50 && !a_success && found_count < target_count; x_tries++) {
                u128 x0r = (u128)rng64(&rng) | ((u128)rng64(&rng) << 64); x0r %= p;
                if (x0r < 2) continue;

                u128 x0m = toM128(x0r, &mt);

                for (int mi=0; mi<nms && !a_success && found_count < target_count; mi++) {
                    u128 QX, QZ;
                    xMUL128(&QX, &QZ, x0m, ms[mi], a24m, &mt);
                    if (QZ == 0) continue;
                    u128 CX=QX, CZ=QZ;
                    int zs = -1;
                    for (int s=1; s<=k+10 && s<50; s++) {
                        xDBL128(&CX,&CZ,CX,CZ,a24m,&mt);
                        if (CZ==0) { zs=s; break; }
                    }
                    if (zs < k) continue;

                    int target_z = zs - k;
                    CX=QX; CZ=QZ;
                    for (int s=0; s<target_z; s++) xDBL128(&CX,&CZ,CX,CZ,a24m,&mt);

                    u128 cz = frM128(CZ, &mt);
                    if (cz == 0) continue;
                    u128 czinv_m = mt.one, base = CZ; u128 e2 = p-2;
                    while (e2>0) { if(e2&1) czinv_m=mm128(czinv_m,base,&mt); base=mm128(base,base,&mt); e2>>=1; }
                    u128 xR = frM128(mm128(CX, czinv_m, &mt), &mt);

                    if (verify128(p, A, xR)) {
#pragma omp critical
                        {
                            int is_dup = 0;
                            for(int j = 0; j < found_count; j++) {
                                if(out_A[j] == A && out_x0[j] == xR) {
                                    is_dup = 1; break;
                                }
                            }
                            if (!is_dup && found_count < target_count) {
                                out_A[found_count] = A;
                                out_x0[found_count] = xR;

                                u64 exact_total_trials = 0;
                                for (int t = 0; t < nthr; t++) {
                                    exact_total_trials += thread_trials[t * 8];
                                }
                                if (exact_total_trials == 0) exact_total_trials = 1;

                                out_trials[found_count] = exact_total_trials;
                                found_count++;
                                a_success = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    return found_count;
}

/* ================================================================
 * Batch-processing main
 * ================================================================ */

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: ./pomerance <input_primes.txt> <output_pure.csv> <output_metrics.csv> <num_triples>\n");
        return 1;
    }

    int target = atoi(argv[4]);
    if (target <= 0) {
        printf("Error: <num_triples> must be greater than 0.\n");
        return 1;
    }

    FILE *input = fopen(argv[1], "r");
    FILE *out_pure = fopen(argv[2], "w");
    FILE *out_metrics = fopen(argv[3], "w");

    if (!input || !out_pure || !out_metrics) {
        printf("Error opening files.\n");
        return 1;
    }

    // Dynamically allocate arrays based on target size
    u64 *out_A_arr = (u64 *)malloc(target * sizeof(u64));
    u64 *out_x0_arr = (u64 *)malloc(target * sizeof(u64));
    u64 *out_trials_arr = (u64 *)malloc(target * sizeof(u64));
    u128 *out_A128 = (u128 *)malloc(target * sizeof(u128));
    u128 *out_x0128 = (u128 *)malloc(target * sizeof(u128));

    if (!out_A_arr || !out_x0_arr || !out_trials_arr || !out_A128 || !out_x0128) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    fprintf(out_metrics, "prime,A,x0,trials\n");

    unsigned long long p_low;
    unsigned long long current_index = 0;

    // Notice we now only expect one parameter per line
    while (fscanf(input, "%llu", &p_low) == 1) {
        current_index++;

        u128 p = (u128)p_low;

        printf("[%llu] Processing prime: %llu (Target: %d)...\n", current_index, p_low, target);
        fflush(stdout);

        int found_amount = 0;

        if (p < ((u128)1 << 63)) {
            found_amount = search64((u64)p, target, out_A_arr, out_x0_arr, out_trials_arr);
        } else {
            found_amount = search128(p, target, out_A128, out_x0128, out_trials_arr);
            for(int i = 0; i < found_amount; i++) {
                out_A_arr[i] = (u64)out_A128[i];
                out_x0_arr[i] = (u64)out_x0128[i];
            }
        }

        if (found_amount > 0) {

            // Sort by timeline mapping (trials)
            for (int i = 0; i < found_amount - 1; i++) {
                for (int j = 0; j < found_amount - i - 1; j++) {
                    if (out_trials_arr[j] > out_trials_arr[j+1]) {
                        u64 temp_t = out_trials_arr[j];
                        out_trials_arr[j] = out_trials_arr[j+1];
                        out_trials_arr[j+1] = temp_t;
                        u64 temp_A = out_A_arr[j];
                        out_A_arr[j] = out_A_arr[j+1];
                        out_A_arr[j+1] = temp_A;
                        u64 temp_x0 = out_x0_arr[j];
                        out_x0_arr[j] = out_x0_arr[j+1];
                        out_x0_arr[j+1] = temp_x0;
                    }
                }
            }

            u64 total_new_trials = 0;
            u64 last_cumulative = 0;

            for (int i = 0; i < found_amount; i++) {
                u64 marginal_trials = out_trials_arr[i] - last_cumulative;
                if (marginal_trials == 0) marginal_trials = 1;

                last_cumulative = out_trials_arr[i];
                total_new_trials += marginal_trials;

                fprintf(out_pure, "%llu,%llu,%llu\n", (unsigned long long)p_low, (unsigned long long)out_A_arr[i], (unsigned long long)out_x0_arr[i]);
                fprintf(out_metrics, "%llu,%llu,%llu,%llu\n", (unsigned long long)p_low, (unsigned long long)out_A_arr[i], (unsigned long long)out_x0_arr[i], marginal_trials);
            }

            printf("[%llu] Success: found %d triples (Total new trials: %llu)\n\n", current_index, found_amount, total_new_trials);
            fflush(stdout);

        } else {
            fprintf(out_pure, "%llu,FAILED,FAILED\n", (unsigned long long)p_low);
            fprintf(out_metrics, "%llu,FAILED,FAILED,FAILED\n", (unsigned long long)p_low);
            printf("Failed: %llu\n\n", (unsigned long long)p_low);
            fflush(stdout);
        }
    }

    // Clean up memory and handles
    free(out_A_arr);
    free(out_x0_arr);
    free(out_trials_arr);
    free(out_A128);
    free(out_x0128);

    fclose(input);
    fclose(out_pure);
    fclose(out_metrics);
    return 0;
}