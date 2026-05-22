// Validation + speed benchmark on an ann-benchmarks-style HDF5 dataset.
//
// Expected HDF5 layout (e.g. SIFT-128, GLOVE, etc.):
//   /train     [N_train, D] float32
//   /test      [N_test,  D] float32
//   /neighbors [N_test, K]   int32  (ground-truth top-K of test against train)
//   /distance  optional, ignored
//
// Recall@K is measured for each bit width vs. the float32 brute-force baseline.
// Speeds: ingest (quant vs none), read (dequant vs none), distance (ADC vs BLAS GEMM).

#include <cblas.h>
#include <hdf5.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include "turboquant/turboquant.h"

namespace {

using turboquant::AdcScore;
using turboquant::PayloadSize;
using turboquant::QuantBits;
using turboquant::Quantize;
using turboquant::Rotator;

struct Dataset {
  size_t dim;
  size_t n_train;
  size_t n_test;
  size_t k_gt;
  std::vector<float> train;        // n_train * dim, row-major
  std::vector<float> test;         // n_test  * dim
  std::vector<int32_t> neighbors;  // n_test  * k_gt
  std::string metric;              // "euclidean" or "angular" (best-effort guess)
};

bool ReadDataset(const std::string& path, Dataset* ds) {
  hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    std::fprintf(stderr, "Failed to open %s\n", path.c_str());
    return false;
  }
  auto read_float = [&](const char* name, std::vector<float>* out,
                        size_t* rows, size_t* cols) {
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
    if (dset < 0) return false;
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    *rows = dims[0];
    *cols = dims[1];
    out->resize(dims[0] * dims[1]);
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out->data());
    H5Sclose(space);
    H5Dclose(dset);
    return true;
  };
  size_t d1, d2;
  if (!read_float("/train", &ds->train, &ds->n_train, &ds->dim)) {
    H5Fclose(file);
    return false;
  }
  if (!read_float("/test", &ds->test, &ds->n_test, &d2)) {
    H5Fclose(file);
    return false;
  }
  // neighbors as int32
  {
    hid_t dset = H5Dopen2(file, "/neighbors", H5P_DEFAULT);
    if (dset < 0) {
      H5Fclose(file);
      return false;
    }
    hid_t space = H5Dget_space(dset);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    ds->neighbors.resize(dims[0] * dims[1]);
    ds->k_gt = dims[1];
    H5Dread(dset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            ds->neighbors.data());
    H5Sclose(space);
    H5Dclose(dset);
  }
  // metric attribute (best-effort): try the HDF5 attribute, then fall back to
  // a filename hint. ann-benchmarks files typically encode the metric in the
  // filename (e.g. *-angular.hdf5, *-euclidean.hdf5, *-ip.hdf5).
  ds->metric = "";
  if (H5Aexists(file, "distance")) {
    hid_t attr = H5Aopen(file, "distance", H5P_DEFAULT);
    hid_t atype = H5Aget_type(attr);
    H5T_class_t cls = H5Tget_class(atype);
    if (cls == H5T_STRING) {
      if (H5Tis_variable_str(atype)) {
        char* p = nullptr;
        H5Aread(attr, atype, &p);
        if (p) {
          ds->metric = p;
          H5free_memory(p);
        }
      } else {
        size_t sz = H5Tget_size(atype);
        std::string s(sz, '\0');
        H5Aread(attr, atype, s.data());
        while (!s.empty() && s.back() == '\0') s.pop_back();
        ds->metric = s;
      }
    }
    H5Tclose(atype);
    H5Aclose(attr);
  }
  if (ds->metric.empty()) {
    if (path.find("angular") != std::string::npos ||
        path.find("cosine") != std::string::npos) {
      ds->metric = "angular";
    } else {
      ds->metric = "euclidean";
    }
  }
  H5Fclose(file);
  (void)d1;
  (void)d2;
  return true;
}

double Now() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

void TopKByScore(const float* scores, size_t n, int k, std::vector<int32_t>* out,
                 bool larger_is_better) {
  // Min/max heap of (score, idx); we want top-k by larger_is_better.
  using Pair = std::pair<float, int32_t>;
  auto cmp_max = [](const Pair& a, const Pair& b) { return a.first > b.first; };  // min-heap of largest
  auto cmp_min = [](const Pair& a, const Pair& b) { return a.first < b.first; };  // max-heap of smallest
  if (larger_is_better) {
    std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_max)> pq(cmp_max);
    for (size_t i = 0; i < n; ++i) {
      if (static_cast<int>(pq.size()) < k) {
        pq.push({scores[i], static_cast<int32_t>(i)});
      } else if (scores[i] > pq.top().first) {
        pq.pop();
        pq.push({scores[i], static_cast<int32_t>(i)});
      }
    }
    out->resize(pq.size());
    for (int j = static_cast<int>(pq.size()) - 1; j >= 0; --j) {
      (*out)[j] = pq.top().second;
      pq.pop();
    }
  } else {
    std::priority_queue<Pair, std::vector<Pair>, decltype(cmp_min)> pq(cmp_min);
    for (size_t i = 0; i < n; ++i) {
      if (static_cast<int>(pq.size()) < k) {
        pq.push({scores[i], static_cast<int32_t>(i)});
      } else if (scores[i] < pq.top().first) {
        pq.pop();
        pq.push({scores[i], static_cast<int32_t>(i)});
      }
    }
    out->resize(pq.size());
    for (int j = static_cast<int>(pq.size()) - 1; j >= 0; --j) {
      (*out)[j] = pq.top().second;
      pq.pop();
    }
  }
}

float Recall(const std::vector<int32_t>& pred, const int32_t* truth, int k) {
  std::unordered_set<int32_t> truth_set(truth, truth + k);
  int hits = 0;
  const int K = std::min<int>(k, static_cast<int>(pred.size()));
  for (int i = 0; i < K; ++i) {
    if (truth_set.count(pred[i])) ++hits;
  }
  return static_cast<float>(hits) / static_cast<float>(k);
}

// L2-normalize rows in place.
void NormalizeRows(float* data, size_t n, size_t d) {
  for (size_t i = 0; i < n; ++i) {
    double s = 0;
    for (size_t j = 0; j < d; ++j) s += data[i * d + j] * data[i * d + j];
    float inv = s > 0 ? 1.0f / static_cast<float>(std::sqrt(s)) : 0.0f;
    for (size_t j = 0; j < d; ++j) data[i * d + j] *= inv;
  }
}

void RunOnDataset(const std::string& path, int k_eval, size_t max_test) {
  Dataset ds;
  if (!ReadDataset(path, &ds)) {
    std::fprintf(stderr, "Could not load %s\n", path.c_str());
    return;
  }
  if (max_test && ds.n_test > max_test) ds.n_test = max_test;
  std::printf("\n=== %s ===\n", path.c_str());
  std::printf("dim=%zu n_train=%zu n_test=%zu k_gt=%zu metric=%s\n", ds.dim,
              ds.n_train, ds.n_test, ds.k_gt, ds.metric.c_str());

  const bool angular = ds.metric == "angular" || ds.metric == "cosine";
  if (angular) {
    // Unit-normalize so dot product == cosine similarity.
    NormalizeRows(ds.train.data(), ds.n_train, ds.dim);
    NormalizeRows(ds.test.data(), ds.n_test, ds.dim);
  }
  const bool larger_is_better = angular;  // cosine: larger is closer; L2: smaller

  // -------------------- Ingest speed --------------------
  std::printf("\n[Ingest: quantize train set]\n");
  std::printf("%-8s %15s %15s\n", "bits", "ms_total", "MB/s_quantized");

  Rotator R(ds.dim, 0x12345);
  const QuantBits bits_list[] = {QuantBits::B1, QuantBits::B2, QuantBits::B4,
                                 QuantBits::B6, QuantBits::B8, QuantBits::B12};
  std::vector<std::vector<uint8_t>> all_payloads(6);
  std::vector<size_t> payload_sizes(6);
  double t_baseline_copy = 0;
  {
    const double t0 = Now();
    std::vector<float> sink(ds.dim);
    for (size_t i = 0; i < ds.n_train; ++i) {
      std::memcpy(sink.data(), ds.train.data() + i * ds.dim,
                  ds.dim * sizeof(float));
    }
    t_baseline_copy = Now() - t0;
    std::printf("%-8s %15.2f %15.2f  (baseline memcpy)\n", "f32",
                1000.0 * t_baseline_copy,
                (ds.n_train * ds.dim * sizeof(float)) / 1e6 / t_baseline_copy);
  }
  for (int bi = 0; bi < 6; ++bi) {
    const QuantBits b = bits_list[bi];
    const size_t ps = PayloadSize(ds.dim, b);
    payload_sizes[bi] = ps;
    all_payloads[bi].assign(ps * ds.n_train, 0);
    const double t0 = Now();
    for (size_t i = 0; i < ds.n_train; ++i) {
      Quantize(R, b, ds.train.data() + i * ds.dim,
               all_payloads[bi].data() + i * ps);
    }
    const double dt = Now() - t0;
    std::printf("%-8d %15.2f %15.2f\n", static_cast<int>(b), 1000.0 * dt,
                (ds.n_train * ps) / 1e6 / dt);
  }

  // -------------------- Read speed (dequantize) --------------------
  std::printf("\n[Read: dequantize one full pass of train set]\n");
  std::printf("%-8s %15s %15s\n", "bits", "ms_total", "MB/s_floats");
  std::printf("%-8s %15.2f %15.2f  (baseline memcpy)\n", "f32",
              1000.0 * t_baseline_copy,
              (ds.n_train * ds.dim * sizeof(float)) / 1e6 / t_baseline_copy);
  for (int bi = 0; bi < 6; ++bi) {
    const QuantBits b = bits_list[bi];
    const size_t ps = payload_sizes[bi];
    std::vector<float> out(ds.dim);
    const double t0 = Now();
    for (size_t i = 0; i < ds.n_train; ++i) {
      turboquant::Dequantize(R, b, all_payloads[bi].data() + i * ps,
                             out.data());
    }
    const double dt = Now() - t0;
    std::printf("%-8d %15.2f %15.2f\n", static_cast<int>(b), 1000.0 * dt,
                (ds.n_train * ds.dim * sizeof(float)) / 1e6 / dt);
  }

  // -------------------- Distance speed --------------------
  // Three paths compared, per bit width:
  //   f32 (BLAS)          : sgemm(test, train^T) on original floats.
  //   ADC                 : per-pair AdcScore against packed codes.
  //   Dequant + BLAS      : dequantize the whole base back to float32 once,
  //                         then sgemm against the dequantized base.
  // The recall column is computed against the float32 BLAS ground truth.
  const size_t Nq = std::min<size_t>(ds.n_test, 1000);
  std::printf("\n[Distance: %zu queries x %zu base vectors]\n", Nq, ds.n_train);

  // ---- float32 BLAS baseline ----
  std::vector<float> blas_scores(Nq * ds.n_train);
  double t_blas = 0;
  {
    const double t0 = Now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(Nq), static_cast<int>(ds.n_train),
                static_cast<int>(ds.dim), 1.0f, ds.test.data(),
                static_cast<int>(ds.dim), ds.train.data(),
                static_cast<int>(ds.dim), 0.0f, blas_scores.data(),
                static_cast<int>(ds.n_train));
    t_blas = Now() - t0;
  }

  // Pre-rotate all queries once (ADC needs them in the rotated space).
  std::vector<std::vector<float>> q_rot(Nq, std::vector<float>(R.padded_dim()));
  for (size_t i = 0; i < Nq; ++i) {
    turboquant::RotateQuery(R, ds.test.data() + i * ds.dim, q_rot[i].data());
  }

  // Precompute base norms (needed for euclidean ranking from a dot-product source).
  std::vector<float> base_norms2;
  if (!angular) {
    base_norms2.resize(ds.n_train);
    for (size_t j = 0; j < ds.n_train; ++j) {
      double s = 0;
      for (size_t d = 0; d < ds.dim; ++d)
        s += ds.train[j * ds.dim + d] * ds.train[j * ds.dim + d];
      base_norms2[j] = static_cast<float>(s);
    }
  }

  // Ground truth top-K from float32 BLAS scores, in the dataset's metric.
  std::vector<std::vector<int32_t>> gt_topk(Nq);
  {
    std::vector<float> tmp(ds.n_train);
    for (size_t i = 0; i < Nq; ++i) {
      const float* row = blas_scores.data() + i * ds.n_train;
      if (angular) {
        TopKByScore(row, ds.n_train, k_eval, &gt_topk[i],
                    /*larger_is_better=*/true);
      } else {
        for (size_t j = 0; j < ds.n_train; ++j) {
          tmp[j] = base_norms2[j] - 2.0f * row[j];
        }
        TopKByScore(tmp.data(), ds.n_train, k_eval, &gt_topk[i],
                    /*larger_is_better=*/false);
      }
    }
  }

  // Header
  std::printf("\n%-14s %12s %12s %10s %12s\n",
              "path", "ms_total", "Mops/s", "recall@K", "rel_vs_f32");
  std::printf("%-14s %12.2f %12.2f %10.4f %12s\n",
              "f32 (BLAS)", 1000.0 * t_blas,
              static_cast<double>(Nq) * ds.n_train / 1e6 / t_blas,
              1.0, "1.00x");

  auto ScoresToRanking = [&](const float* dots, std::vector<float>* out,
                             const float* base_norm2_for_dequant) {
    out->resize(ds.n_train);
    if (angular) {
      for (size_t j = 0; j < ds.n_train; ++j) (*out)[j] = dots[j];
    } else {
      // Use the appropriate base norms (original for f32 baseline; dequant norms
      // when scoring against dequantized base).
      const float* nrm = base_norm2_for_dequant ? base_norm2_for_dequant
                                                : base_norms2.data();
      for (size_t j = 0; j < ds.n_train; ++j) {
        (*out)[j] = nrm[j] - 2.0f * dots[j];
      }
    }
  };

  std::vector<float> dequant_buf;  // reusable scratch

  for (int bi = 0; bi < 6; ++bi) {
    const QuantBits b = bits_list[bi];
    const size_t ps = payload_sizes[bi];

    // -------- ADC path --------
    double adc_recall_sum = 0;
    const double t_adc0 = Now();
    {
      std::vector<float> scores(ds.n_train);
      for (size_t i = 0; i < Nq; ++i) {
        const float* qr = q_rot[i].data();
        for (size_t j = 0; j < ds.n_train; ++j) {
          auto s = AdcScore(R, b, qr, all_payloads[bi].data() + j * ps);
          scores[j] = angular ? s.dot : (s.decoded_norm2 - 2.0f * s.dot);
        }
        std::vector<int32_t> top;
        TopKByScore(scores.data(), ds.n_train, k_eval, &top,
                    /*larger_is_better=*/angular ? true : false);
        adc_recall_sum += Recall(top, gt_topk[i].data(), k_eval);
      }
    }
    const double t_adc = Now() - t_adc0;

    // -------- Dequant + BLAS path --------
    // Step 1: dequantize the entire base into a row-major float32 matrix.
    dequant_buf.assign(ds.n_train * ds.dim, 0.0f);
    const double t_deq0 = Now();
    for (size_t j = 0; j < ds.n_train; ++j) {
      turboquant::Dequantize(R, b, all_payloads[bi].data() + j * ps,
                             dequant_buf.data() + j * ds.dim);
    }
    const double t_deq = Now() - t_deq0;

    // Step 2: BLAS sgemm queries x dequantized^T.
    std::vector<float> deq_dots(Nq * ds.n_train);
    const double t_gemm0 = Now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(Nq), static_cast<int>(ds.n_train),
                static_cast<int>(ds.dim), 1.0f, ds.test.data(),
                static_cast<int>(ds.dim), dequant_buf.data(),
                static_cast<int>(ds.dim), 0.0f, deq_dots.data(),
                static_cast<int>(ds.n_train));
    const double t_gemm = Now() - t_gemm0;

    // For euclidean ranking on dequantized base we need its row norms.
    std::vector<float> deq_norms2;
    if (!angular) {
      deq_norms2.resize(ds.n_train);
      for (size_t j = 0; j < ds.n_train; ++j) {
        double s = 0;
        for (size_t d = 0; d < ds.dim; ++d) {
          const float v = dequant_buf[j * ds.dim + d];
          s += v * v;
        }
        deq_norms2[j] = static_cast<float>(s);
      }
    }

    double deq_recall_sum = 0;
    {
      std::vector<float> ranked;
      for (size_t i = 0; i < Nq; ++i) {
        ScoresToRanking(deq_dots.data() + i * ds.n_train, &ranked,
                        angular ? nullptr : deq_norms2.data());
        std::vector<int32_t> top;
        TopKByScore(ranked.data(), ds.n_train, k_eval, &top,
                    /*larger_is_better=*/angular ? true : false);
        deq_recall_sum += Recall(top, gt_topk[i].data(), k_eval);
      }
    }
    const double t_deqblas = t_deq + t_gemm;

    const double ops = static_cast<double>(Nq) * ds.n_train;
    char label_adc[32], label_deq[32];
    std::snprintf(label_adc, sizeof(label_adc), "b%-2d (ADC)",
                  static_cast<int>(b));
    std::snprintf(label_deq, sizeof(label_deq), "b%-2d (deq+BLAS)",
                  static_cast<int>(b));
    char rel_adc[16], rel_deq[16];
    std::snprintf(rel_adc, sizeof(rel_adc), "%.2fx", t_adc / t_blas);
    std::snprintf(rel_deq, sizeof(rel_deq), "%.2fx", t_deqblas / t_blas);

    std::printf("%-14s %12.2f %12.2f %10.4f %12s\n", label_adc,
                1000.0 * t_adc, ops / 1e6 / t_adc,
                adc_recall_sum / Nq, rel_adc);
    std::printf("%-14s %12.2f %12.2f %10.4f %12s  (%.0fms deq + %.0fms gemm)\n",
                label_deq, 1000.0 * t_deqblas, ops / 1e6 / t_deqblas,
                deq_recall_sum / Nq, rel_deq,
                1000.0 * t_deq, 1000.0 * t_gemm);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "Usage: %s <dataset.hdf5> [k_eval=10] [max_test=1000]\n",
                 argv[0]);
    return 1;
  }
  const std::string path = argv[1];
  const int k_eval = argc > 2 ? std::atoi(argv[2]) : 10;
  const size_t max_test = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1000;
  RunOnDataset(path, k_eval, max_test);
  return 0;
}
