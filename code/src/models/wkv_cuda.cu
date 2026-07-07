#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>

#define MIN_VALUE (-1e38f)
#define EPS (1e-6f)

#define CHANNEL_SPLIT (256 / 16)
#define TOKEN_SPLIT   (256 / CHANNEL_SPLIT)


#ifndef Tmax
#define Tmax 4096
#endif

#define IDEAL_T_LEN (Tmax / TOKEN_SPLIT)

__device__ __forceinline__ float fmax2(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ int imin2(int a, int b) { return a < b ? a : b; }


__global__ void kernel_forward(const int B, const int T, const int C,
                               const float *__restrict__ const _w,
                               const float *__restrict__ const _u,
                               const float *__restrict__ const _k,
                               const float *__restrict__ const _v,
                               float *__restrict__ const _y) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel_id = threadIdx.x;
    const int token_id = threadIdx.y;
    const int _b = idx / C;
    const int _c = idx % C;

    const int _T = (T + TOKEN_SPLIT - 1) / TOKEN_SPLIT;
    const int _t = _T * token_id;
    const int _offset = _b * T * C + _c;
    const int _tokenLength = imin2(T - _t, _T);

    float u = _u[_c];
    float w = _w[_c];

    const float *__restrict__ const k = _k + _offset;
    const float *__restrict__ const v = _v + _offset;
    float *__restrict__ const y = _y + _offset;

    __shared__ float Sa[CHANNEL_SPLIT][TOKEN_SPLIT];
    __shared__ float Sb[CHANNEL_SPLIT][TOKEN_SPLIT];
    __shared__ float So2[CHANNEL_SPLIT][TOKEN_SPLIT];

    float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
    float o1 = MIN_VALUE, o2 = MIN_VALUE;

    for (int i = _t; i < (_t + _tokenLength); i++){
        const int ii = i * C;
        float no = fmax2(o1, k[ii] - w * (float)(i - _t));
        float e1 = expf(o1 - no);
        float e3 = expf(k[ii] - w * (float)(i - _t) - no);
        c = e1 * c + e3 * v[ii];
        d = e1 * d + e3;
        o1 = no;

        const int ni = 2 * _t + _tokenLength - 1 - i;
        const int nini = ni * C;
        const int exp_w = _t + _tokenLength - ni;
        no = fmax2(o2, k[nini] - w * (float)exp_w);
        float e2 = expf(o2 - no);
        e3 = expf(k[nini] - w * (float)exp_w - no);
        a = e2 * a + e3 * v[nini];
        b = e2 * b + e3;
        o2 = no;
    }

    So2[channel_id][token_id] = o2;
    Sa[channel_id][token_id] = a;
    Sb[channel_id][token_id] = b;
    __syncthreads();

    a = 0.f;
    b = 0.f;
    o2 = MIN_VALUE;
    for (int i = 0; i < token_id; i++){
        const int exp_w = (token_id - i - 1) * _T;
        float no = fmax2(So2[channel_id][i] - w * (float)exp_w, o2);
        a = a * expf(o2 - no) + Sa[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - no);
        b = b * expf(o2 - no) + Sb[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - no);
        o2 = no;
    }
    __syncthreads();

    Sa[channel_id][token_id] = c;
    Sb[channel_id][token_id] = d;
    So2[channel_id][token_id] = o1;
    __syncthreads();

    c = 0.f;
    d = 0.f;
    o1 = MIN_VALUE;
    for (int i = token_id; i < TOKEN_SPLIT; i++){
        const int exp_w = (i - token_id) * _T;
        float no = fmax2(So2[channel_id][i] - w * (float)exp_w, o1);
        c = c * expf(o1 - no) + Sa[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - no);
        d = d * expf(o1 - no) + Sb[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - no);
        o1 = no;
    }
    c -= expf(k[_t * C] - o1) * v[_t * C];
    d -= expf(k[_t * C] - o1);

    for (int i = _t; i < (_t + _tokenLength); i++) {
        const int ii = i * C;
        float no = fmax2(o1, u + k[ii]);
        no = fmax2(no, o2);
        float e1 = expf(o1 - no);
        float e2 = expf(o2 - no);
        float e3 = expf(u + k[ii] - no);
        y[ii] = (c * e1 + a * e2 + e3 * v[ii]) / (d * e1 + b * e2 + e3 + EPS);

        const int ii2 = ((i + 1) % T) * C;
        no = fmax2(o2 - w, k[ii]);
        e2 = expf(o2 - w - no);
        e3 = expf(k[ii] - no);
        a = e2 * a + e3 * v[ii];
        b = e2 * b + e3;
        o2 = no;

        no = fmax2(o1 + w, k[ii2] + w);
        e1 = expf(o1 + w - no);
        e3 = expf(k[ii2] + w - no);
        c = e1 * c - e3 * v[ii2];
        d = e1 * d - e3;
        o1 = no;
    }
}


__global__ void kernel_backward(const int B, const int T, const int C,
                                const float *__restrict__ const _w,
                                const float *__restrict__ const _u,
                                const float *__restrict__ const _k,
                                const float *__restrict__ const _v,
                                const float *__restrict__ const _gy,
                                float *__restrict__ const _gw,
                                float *__restrict__ const _gu,
                                float *__restrict__ const _gk,
                                float *__restrict__ const _gv) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel_id = threadIdx.x;
    const int token_id = threadIdx.y;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _T = (T + TOKEN_SPLIT - 1) / TOKEN_SPLIT;
    const int _t = _T * token_id;
    const int _offset = _b * T * C + _c;
    const int _tokenLength = imin2(T - _t, _T);

    float u = _u[_c];
    float w = _w[_c];

    const float *__restrict__ const k = _k + _offset;
    const float *__restrict__ const v = _v + _offset;
    const float *__restrict__ const gy = _gy + _offset;

    float *__restrict__ const gk = _gk + _offset;
    float *__restrict__ const gv = _gv + _offset;

    float y[IDEAL_T_LEN], z[IDEAL_T_LEN], zexp[IDEAL_T_LEN];

    __shared__ float Sa[CHANNEL_SPLIT][TOKEN_SPLIT];
    __shared__ float Sb[CHANNEL_SPLIT][TOKEN_SPLIT];
    __shared__ float Sdadw[CHANNEL_SPLIT][TOKEN_SPLIT];
    __shared__ float Sdbdw[CHANNEL_SPLIT][TOKEN_SPLIT];
    __shared__ float So2[CHANNEL_SPLIT][TOKEN_SPLIT];

    float a = 0.f, b = 0.f, c = 0.f, d = 0.f;
    float dadw = 0.f, dbdw = 0.f, dcdw = 0.f, dddw = 0.f;
    float o1 = MIN_VALUE, o2 = MIN_VALUE;

    for (int i = _t; i < (_t + _tokenLength); i++){
        const int ii = i * C;
        float no = fmax2(o1, k[ii] - w * (float)(i - _t));
        float e1 = expf(o1 - no);
        float e3 = expf(k[ii] - w * (float)(i - _t) - no);
        dcdw = dcdw * e1 - (float)(i - _t) * e3 * v[ii];
        dddw = dddw * e1 - (float)(i - _t) * e3;
        c = e1 * c + e3 * v[ii];
        d = e1 * d + e3;
        o1 = no;

        const int ni = 2 * _t + _tokenLength - 1 - i;
        const int nini = ni * C;
        const int exp_w = _t + _tokenLength - ni;
        no = fmax2(o2, k[nini] - w * (float)exp_w);
        float e2 = expf(o2 - no);
        e3 = expf(k[nini] - w * (float)exp_w - no);
        dadw = dadw * e2 - (float)exp_w * e3 * v[nini];
        dbdw = dbdw * e2 - (float)exp_w * e3;
        a = e2 * a + e3 * v[nini];
        b = e2 * b + e3;
        o2 = no;
    }

    __syncthreads();
    So2[channel_id][token_id] = o2;
    Sa[channel_id][token_id] = a;
    Sb[channel_id][token_id] = b;
    Sdadw[channel_id][token_id] = dadw;
    Sdbdw[channel_id][token_id] = dbdw;
    __syncthreads();

    a = 0.f; b = 0.f; dadw = 0.f; dbdw = 0.f;
    o2 = MIN_VALUE;
    for (int i = 0; i < token_id; i++){
        const int exp_w = (token_id - i - 1) * _T;
        float no = fmax2(So2[channel_id][i] - w * (float)exp_w, o2);
        a = a * expf(o2 - no) + Sa[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - no);
        b = b * expf(o2 - no) + Sb[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - no);
        dadw = dadw * expf(o2 - no) + (Sdadw[channel_id][i] - (float)exp_w * Sa[channel_id][i])
            * expf(So2[channel_id][i] - w * (float)exp_w - no);
        dbdw = dbdw * expf(o2 - no) + (Sdbdw[channel_id][i] - (float)exp_w * Sb[channel_id][i])
            * expf(So2[channel_id][i] - w * (float)exp_w - no);
        o2 = no;
    }

    __syncthreads();
    So2[channel_id][token_id] = o1;
    Sa[channel_id][token_id] = c;
    Sb[channel_id][token_id] = d;
    Sdadw[channel_id][token_id] = dcdw;
    Sdbdw[channel_id][token_id] = dddw;
    __syncthreads();

    c = 0.f; d = 0.f; dcdw = 0.f; dddw = 0.f;
    o1 = MIN_VALUE;
    for (int i = token_id; i < TOKEN_SPLIT; i++){
        const int exp_w = (i - token_id) * _T;
        float no = fmax2(So2[channel_id][i] - w * (float)exp_w, o1);
        c = c * expf(o1 - no) + Sa[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - no);
        d = d * expf(o1 - no) + Sb[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - no);
        dcdw = dcdw * expf(o1 - no) + (Sdadw[channel_id][i] - (float)exp_w * Sa[channel_id][i])
             * expf(So2[channel_id][i] - w * (float)exp_w - no);
        dddw = dddw * expf(o1 - no) + (Sdbdw[channel_id][i] - (float)exp_w * Sb[channel_id][i])
             * expf(So2[channel_id][i] - w * (float)exp_w - no);
        o1 = no;
    }

    c -= expf(k[_t * C] - o1) * v[_t * C];
    d -= expf(k[_t * C] - o1);

    float gw = 0.f, gu = 0.f;
    float gc = 0.f, gd = 0.f, ga = 0.f, gb = 0.f;
    float go1 = MIN_VALUE, go2 = MIN_VALUE;

    for (int i = _t; i < (_t + _tokenLength); i++) {
        const int ii = i * C;
        float no = fmax2(o1, u + k[ii]);
        no = fmax2(no, o2);
        float e1 = expf(o1 - no);
        float e2 = expf(o2 - no);
        float e3 = expf(u + k[ii] - no);
        float num = (c * e1 + a * e2 + e3 * v[ii]);
        float iden = 1.f / (d * e1 + b * e2 + e3 + EPS);
        y[i - _t] = num * iden;
        z[i - _t] = iden;
        zexp[i - _t] = -no;

        gw += gy[ii] * (dadw - dbdw * y[i - _t]) * iden * e2;
        gw += gy[ii] * (dcdw - dddw * y[i - _t]) * iden * e1;
        gu += gy[ii] * (v[ii] - y[i - _t]) * e3 * iden;
        gk[ii] = gy[ii] * iden * (v[ii] - y[i - _t]) * e3;
        gv[ii] = gy[ii] * iden * e3;

        float gno = fmax2(-w + go1, -no);
        e1 = expf(-w + go1 - gno);
        e3 = gy[ii] * iden  * expf(-no - gno);
        gc = e1 * gc + e3 * y[i - _t];
        gd = e1 * gd + e3;
        go1 = gno;

        const int ii2 = ((i + 1) % T) * C;
        no = fmax2(o2 - w, k[ii]);
        e2 = expf(o2 - w - no);
        e3 = expf(k[ii] - no);
        dadw = e2 * (dadw - a);
        dbdw = e2 * (dbdw - b);
        a = e2 * a + e3 * v[ii];
        b = e2 * b + e3;
        o2 = no;

        no = fmax2(o1 + w, k[ii2] + w);
        e1 = expf(o1 + w - no);
        e3 = expf(k[ii2] + w - no);
        dcdw = e1 * (c + dcdw) - e3 * v[ii2];
        dddw = e1 * (d + dddw) - e3;
        c = e1 * c - e3 * v[ii2];
        d = e1 * d - e3;
        o1 = no;
    }

    __syncthreads();
    Sdadw[channel_id][token_id] = gw;
    Sdbdw[channel_id][token_id] = gu;
    __syncthreads();

    if(token_id == 0){
        const int _offsetBC = _b * C + _c;
        for(int i = 0; i < TOKEN_SPLIT; i++){
            _gw[_offsetBC] += Sdadw[channel_id][i];
            _gu[_offsetBC] += Sdbdw[channel_id][i];
        }
    }
    __syncthreads();

    for (int i = _t + _tokenLength - 1; i >=_t ; i--) {
        const int ii = i * C;
        float gno = fmax2(-w + go2, zexp[i - _t]);
        float e2 = expf(-w + go2 - gno);
        float e3 = gy[ii] * z[i - _t] * expf(zexp[i - _t] - gno);
        ga = e2 * ga + e3 * y[i - _t];
        gb = e2 * gb + e3;
        go2 = gno;
    }

    __syncthreads();
    Sa[channel_id][token_id] = gc;
    Sb[channel_id][token_id] = gd;
    So2[channel_id][token_id] = go1;
    __syncthreads();

    gc = 0.f; gd = 0.f; go1 = MIN_VALUE;
    for (int i = 0; i < token_id; i++){
        const int exp_w = (token_id - i - 1) * _T;
        float gno = fmax2(So2[channel_id][i] - w * (float)exp_w, go1);
        gc = gc * expf(go1 - gno) + Sa[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - gno);
        gd = gd * expf(go1 - gno) + Sb[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - gno);
        go1 = gno;
    }

    __syncthreads();
    Sa[channel_id][token_id] = ga;
    Sb[channel_id][token_id] = gb;
    So2[channel_id][token_id] = go2;
    __syncthreads();

    ga = 0.f; gb = 0.f; go2 = MIN_VALUE;
    for (int i = token_id + 1; i < TOKEN_SPLIT; i++){
        const int exp_w = (i - token_id - 1) * _T;
        float gno = fmax2(So2[channel_id][i] - w * (float)exp_w, go2);
        ga = ga * expf(go2 - gno) + Sa[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - gno);
        gb = gb * expf(go2 - gno) + Sb[channel_id][i] * expf(So2[channel_id][i] - w * (float)exp_w - gno);
        go2 = gno;
    }

    for (int i = _t; i < (_t + _tokenLength); i++) {
        const int ii = i * C;
        const int ni = 2 * _t + _tokenLength - 1 - i;
        const int nini = ni * C;

        gk[ii] += expf(k[ii] + go1) * (gd * v[ii] - gc);
        gk[nini] += expf(k[nini] + go2) * (gb * v[nini] - ga);
        gv[ii] += expf(k[ii] + go1) * gd;
        gv[nini] += expf(k[nini] + go2) * gb;

        float gno = fmax2(-w + go1, zexp[i - _t]);
        float e1 = expf(-w + go1 - gno);
        float e3 = gy[ii] * z[i - _t]  * expf(zexp[i - _t] - gno);
        gc = e1 * gc + e3 * y[i - _t];
        gd = e1 * gd + e3;
        go1 = gno;

        gno = fmax2(-w + go2, zexp[ni - _t]);
        float e2 = expf(-w + go2 - gno);
        e3 = gy[nini] * z[ni - _t] * expf(zexp[ni - _t] - gno);
        ga = e2 * ga + e3 * y[ni - _t];
        gb = e2 * gb + e3;
        go2 = gno;
    }
}


extern "C" void cuda_forward(int B, int T, int C,
                             float *w, float *u, float *k, float *v, float *y) {
    dim3 threadsPerBlock(min(CHANNEL_SPLIT, C), TOKEN_SPLIT);
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}

extern "C" void cuda_backward(int B, int T, int C,
                              float *w, float *u, float *k, float *v, float *gy,
                              float *gw, float *gu, float *gk, float *gv) {
    dim3 threadsPerBlock(min(CHANNEL_SPLIT, C), TOKEN_SPLIT);
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_backward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, gy, gw, gu, gk, gv);
}
