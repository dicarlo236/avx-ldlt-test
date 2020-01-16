#include <random>
#include <vector>
#include <cstdio>

#include "Timer.h"
#include "reference_implementation.h"
#include "avx_implementation.h"

struct Problem {
  int n;
  std::vector<double> inv_D;
  std::vector<double> L_values;
  std::vector<int> col_offsets;
  std::vector<int> row_idx;
  std::vector<double> x;
  int nnz;

  void print() {

  }
};


Problem generate_random_problem(int n, double fill) {
  Problem p;
  p.n = n;

  std::mt19937 mt(1212);
  std::uniform_real_distribution<> fill_dist;

  std::uniform_real_distribution<> value_dist(-1, 1); // ??

  // random D, x values.
  for(int i = 0; i < n; i++) {
    p.inv_D.push_back(1.); // ??
    p.x.push_back(value_dist(mt));
  }

  // loop over columns
  int idx = 0;
  for(int col = 0; col < n - 1; col++) {
    // insert start:
    p.col_offsets.push_back(idx);

    for(int row = col+1; row < n; row++) {
      if(fill_dist(mt) > fill) continue;
      p.L_values.push_back(value_dist(mt));
      p.row_idx.push_back(row);
      assert(p.row_idx.back() >= col+1);
      idx++;
    }



  }

  p.col_offsets.push_back(idx);

  p.nnz = idx;

  return p;
}



int main() {
  Problem prob = generate_random_problem(1000, 0.3);
  printf("Problem (n = %d, nnz = %d fill %.3f)\n", prob.n, prob.nnz, (double)prob.nnz/ (prob.n * prob.n));


  // first solve with reference solver
  std::vector<double> x_ref = prob.x;
  Timer timer;
  ref_solve(prob.n, prob.col_offsets.data(), prob.row_idx.data(), prob.L_values.data(), prob.inv_D.data(), x_ref.data());
  auto time = timer.getUs();
  printf("reference solved in %.2f us\n", time);




  // next solve with avx solver
  std::vector<double> x_avx = prob.x;
  timer.start();
  avx_solve(prob.n, prob.col_offsets.data(), prob.row_idx.data(), prob.L_values.data(), prob.inv_D.data(), x_avx.data());
  time = timer.getUs();
  printf("avx solved in %.2f us\n", time);



  bool bad = false;

  for(int i = 0; i < x_ref.size(); i++) {
    double err = std::abs(x_avx[i] - x_ref[i]);
    if(err / std::abs(std::min(x_avx[i], x_ref[i])) > 0.0000000001) {
      bad = true;
      printf("err %.3f\n", err);
    }
  }

  if(bad) {
    printf("FAIL\n");
  } else {
    printf("pass\n");
  }

  return 0;
}