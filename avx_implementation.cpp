#include "avx_implementation.h"
#include <immintrin.h>
#include <cstdio>

static void spTriAVX(int n, const int* p, const int* i, const double* v, double* x)
{
  constexpr int UNROLL = 1;

  for(int c = 0; c < n; c++) {

    __m256d xV = _mm256_set1_pd(x[c]);     // xV = {x[c], x[c], x[c], x[c]}

    int a = p[c];
    int end = p[c+1];
    int vEnd = end - 4 * UNROLL;
    const double* matPtr = v + a;
    const int* iPtr = i + a;
    for(; a < vEnd; a += 4 * UNROLL) {
      for(int u = 0; u < UNROLL; u++) {
        __m256d mat = _mm256_loadu_pd(matPtr);
        __m256d lhs = _mm256_set_pd(x[iPtr[3]], x[iPtr[2]], x[iPtr[1]], x[iPtr[0]]);  // gather?
        __m256d fma = _mm256_fnmadd_pd(xV, mat, lhs);
        //__m256d fma = _mm256_mul_pd(xV, mat);

        x[iPtr[0]] = fma[0];
        x[iPtr[1]] = fma[1];
        x[iPtr[2]] = fma[2];
        x[iPtr[3]] = fma[3];

        matPtr += 4;
        iPtr += 4;
      }
    }

    for(; a < end; a++) {
      x[iPtr[0]] -= matPtr[0] * x[c];
      matPtr++;
      iPtr++;
    }

  }
}

static void spTriTAVX2(int n, const int* p, const int* i, const double* v, double* x) {
  constexpr int UNROLL = 1;
  for(int c = n - 1; c >= 0; c--) { // loop backward over columns
    int colStart = p[c]; // start idx of column
    int end = p[c+1];    // end idx of column
    int vEnd = end - 4 * UNROLL; // end idx of vectorized/unrolled
    const int* iPtr = i + colStart; // indices

    const double* matPtr = v + colStart; // matrix
    __m256d acc = _mm256_set1_pd(0.);

    for(; colStart < vEnd; colStart += 4 * UNROLL) {
      for(int u = 0; u < UNROLL; u++) {
        __m256d mat = _mm256_loadu_pd(matPtr);
        __m256d rhs = _mm256_set_pd(x[iPtr[3]], x[iPtr[2]], x[iPtr[1]], x[iPtr[0]]);
        acc = _mm256_fmadd_pd(mat, rhs, acc);

        matPtr += 4;
        iPtr += 4;
      }
    } // end for

    // horizontal sum and set x
    x[c] -= acc[0];
    x[c] -= acc[1];
    x[c] -= acc[2];
    x[c] -= acc[3];

    // whatever didn't fit in the vectors
    for(;colStart < end; colStart++) {
      x[c] -= *(matPtr++) * x[*(iPtr++)];
    }
  }
}


void avx_solve(int n, int* Lp, int* Li, double* Lx, double* Dinv, double* x) {

  spTriAVX(n, Lp, Li, Lx, x);
//  for(int j = 0; j < n; j++) {
//    int end = Lp[j + 1];
//    for(int p = Lp[j]; p < end; p++) {
//      x[Li[p]] -= Lx[p] * x[j];
//    }
//  }

  for(int i = 0; i < n; i++) {
    x[i] *= Dinv[i];
  }

  spTriTAVX2(n, Lp, Li, Lx, x);

//  for(int i = n - 1; i >= 0; i--) {
//    int end = Lp[i + 1];
//    for(int j = Lp[i]; j < end; j++) {
//      x[i] -= Lx[j] * x[Li[j]];
//    }
//  }
}