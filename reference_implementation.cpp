#include "reference_implementation.h"


void spTri_ref(int n, int*  Lp, int* Li, double* Lx, double* x) {
  for(int i = 0; i < n; i++){
    for(int j = Lp[i]; j < Lp[i+1]; j++){
      x[Li[j]] -= Lx[j] * x[i];
    }
  }
}


void spTriT_ref(int n, int*  Lp, int* Li, double* Lx, double* x) {
  for(int i = n-1; i >= 0; i--){
    for(int j = Lp[i]; j < Lp[i+1]; j++){
      x[i] -= Lx[j] * x[Li[j]];
    }
  }
}

void ref_solve(int n, int* Lp, int* Li, double* Lx, double* Dinv, double* x) {
  spTri_ref(n,Lp,Li,Lx,x);
  for(int i = 0; i < n; i++) x[i] *= Dinv[i];
  spTriT_ref(n,Lp,Li,Lx,x);
}