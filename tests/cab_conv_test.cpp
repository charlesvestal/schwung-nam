/*
 * cab_conv_test - the FFT cab convolver must match direct convolution.
 *
 * Reference: the direct time-domain loop the plugin used before (copied
 * verbatim in spirit: circular history, one MAC per tap per sample), run
 * in double precision. Checks impulse and noise inputs across IR lengths
 * that straddle the partition boundary, then benches both paths.
 *
 *   c++ -O2 -std=c++17 -Isrc/dsp tests/cab_conv_test.cpp src/dsp/pffft/pffft.c -o cab_conv_test
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "cab_conv.h"

static float frand(unsigned *s) {
    *s = *s * 1664525u + 1013904223u;
    return ((*s >> 8) / 8388608.0f) - 1.0f;
}

/* Direct convolution over the whole signal, double accumulator. */
static std::vector<float> direct(const std::vector<float> &x, const std::vector<float> &ir) {
    std::vector<float> y(x.size());
    for (size_t n = 0; n < x.size(); n++) {
        double s = 0;
        for (size_t k = 0; k < ir.size() && k <= n; k++) s += (double)ir[k] * x[n - k];
        y[n] = (float)s;
    }
    return y;
}

/* The plugin's previous apply_cab_ir, for the bench. */
static void old_direct_block(const float *ir, int ir_len, float *hist, int *pos_io, float *audio) {
    const int hist_len = ir_len + CAB_BLOCK;
    int pos = *pos_io;
    for (int i = 0; i < CAB_BLOCK; i++) {
        hist[pos] = audio[i];
        float sum = 0.0f;
        int p = pos;
        for (int k = 0; k < ir_len; k++) {
            sum += ir[k] * hist[p];
            if (--p < 0) p = hist_len - 1;
        }
        audio[i] = sum;
        if (++pos >= hist_len) pos = 0;
    }
    *pos_io = pos;
}

static double now_us() {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main() {
    int fails = 0;
    const int lens[] = {1, 2, 127, 128, 129, 255, 256, 1000, 4096, 8191, 8192};
    unsigned seed = 12345;

    for (int L : lens) {
        std::vector<float> ir(L);
        for (int i = 0; i < L; i++) ir[i] = frand(&seed) * expf(-i / 2000.0f);

        const int blocks = (L + 3 * CAB_BLOCK) / CAB_BLOCK + 4;
        for (int kind = 0; kind < 2; kind++) {
            std::vector<float> x(blocks * CAB_BLOCK, 0.0f);
            if (kind == 0) x[5] = 1.0f;  /* impulse, off the block edge */
            else for (auto &v : x) v = frand(&seed) * 0.5f;

            std::vector<float> ref = direct(x, ir);
            cab_conv_t *c = cab_conv_create(ir.data(), L);
            if (!c) { printf("FAIL create L=%d\n", L); fails++; continue; }
            std::vector<float> y = x;
            for (int b = 0; b < blocks; b++) cab_conv_process(c, y.data() + b * CAB_BLOCK);
            cab_conv_free(c);

            double peak = 0, err = 0;
            for (size_t i = 0; i < y.size(); i++) {
                peak = fmax(peak, fabs(ref[i]));
                err = fmax(err, fabs(ref[i] - y[i]));
            }
            double rel = peak > 0 ? err / peak : err;
            bool ok = rel < 1e-5;
            if (!ok) fails++;
            printf("%s L=%-5d %-7s max|err|=%.3g  rel=%.3g\n", ok ? "ok  " : "FAIL",
                   L, kind ? "noise" : "impulse", err, rel);
        }
    }

    /* Bench: 8192-tap IR, µs per 128-sample block. */
    {
        const int L = 8192, N = 2000;
        std::vector<float> ir(L), buf(CAB_BLOCK), hist(L + CAB_BLOCK, 0.0f);
        for (auto &v : ir) v = frand(&seed) * 0.01f;
        int pos = 0;
        for (auto &v : buf) v = frand(&seed);
        double t0 = now_us();
        for (int i = 0; i < N / 10; i++) old_direct_block(ir.data(), L, hist.data(), &pos, buf.data());
        double t_old = (now_us() - t0) / (N / 10);
        cab_conv_t *c = cab_conv_create(ir.data(), L);
        t0 = now_us();
        for (int i = 0; i < N; i++) cab_conv_process(c, buf.data());
        double t_new = (now_us() - t0) / N;
        cab_conv_free(c);
        printf("bench L=8192: direct %.1f us/block, fft %.1f us/block (%.0fx)\n",
               t_old, t_new, t_old / t_new);
    }

    printf(fails ? "\n%d FAILED\n" : "\nall passed\n", fails);
    return fails ? 1 : 0;
}
