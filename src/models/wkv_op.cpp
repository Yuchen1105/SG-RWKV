#include <torch/extension.h>

extern "C" void cuda_forward(int B, int T, int C,
                             float *w, float *u, float *k, float *v, float *y);
extern "C" void cuda_backward(int B, int T, int C,
                              float *w, float *u, float *k, float *v, float *gy,
                              float *gw, float *gu, float *gk, float *gv);

static inline void check_cuda_f32_contig(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, name, " must be float32");
}

void forward(int64_t B, int64_t T, int64_t C,
             torch::Tensor w, torch::Tensor u,
             torch::Tensor k, torch::Tensor v,
             torch::Tensor y) {
    check_cuda_f32_contig(w, "w");
    check_cuda_f32_contig(u, "u");
    check_cuda_f32_contig(k, "k");
    check_cuda_f32_contig(v, "v");
    check_cuda_f32_contig(y, "y");

    cuda_forward((int)B, (int)T, (int)C,
                 w.data_ptr<float>(),
                 u.data_ptr<float>(),
                 k.data_ptr<float>(),
                 v.data_ptr<float>(),
                 y.data_ptr<float>());
}

void backward(int64_t B, int64_t T, int64_t C,
              torch::Tensor w, torch::Tensor u,
              torch::Tensor k, torch::Tensor v,
              torch::Tensor gy,
              torch::Tensor gw, torch::Tensor gu,
              torch::Tensor gk, torch::Tensor gv) {
    check_cuda_f32_contig(w, "w");
    check_cuda_f32_contig(u, "u");
    check_cuda_f32_contig(k, "k");
    check_cuda_f32_contig(v, "v");
    check_cuda_f32_contig(gy, "gy");
    check_cuda_f32_contig(gw, "gw");
    check_cuda_f32_contig(gu, "gu");
    check_cuda_f32_contig(gk, "gk");
    check_cuda_f32_contig(gv, "gv");

    cuda_backward((int)B, (int)T, (int)C,
                  w.data_ptr<float>(),
                  u.data_ptr<float>(),
                  k.data_ptr<float>(),
                  v.data_ptr<float>(),
                  gy.data_ptr<float>(),
                  gw.data_ptr<float>(),
                  gu.data_ptr<float>(),
                  gk.data_ptr<float>(),
                  gv.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv forward (CUDA)");
    m.def("backward", &backward, "wkv backward (CUDA)");
}
