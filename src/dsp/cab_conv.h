/*
 * Cabinet IR convolution - uniformly partitioned FFT convolution
 * (overlap-save, frequency-domain delay line).
 *
 * The IR is cut into partitions of one block (CAB_BLOCK samples); each
 * partition's spectrum is computed once at load time. Per block the input
 * is transformed once, multiplied against every partition's spectrum with
 * the matching input spectrum from that many blocks ago, and transformed
 * back. Partition length == block length, so there is no added latency.
 *
 * Cost per block is two 2*CAB_BLOCK-point real FFTs plus one complex
 * multiply-accumulate per bin per partition - for an 8192-sample IR,
 * roughly 1/20 of the direct time-domain loop it replaces.
 *
 * FFT: pffft (Julien Pommier, FFTPACK licence), which uses NEON on aarch64.
 */

#ifndef NAM_CAB_CONV_H
#define NAM_CAB_CONV_H

#include <cstring>
#include "pffft/pffft.h"

#define CAB_BLOCK 128
#define CAB_FFT_N (2 * CAB_BLOCK)

typedef struct {
    int parts;         /* number of IR partitions */
    int fdl_pos;       /* slot holding the newest input spectrum */
    float *ir_spec;    /* parts * CAB_FFT_N, pffft internal order */
    float *fdl;        /* parts * CAB_FFT_N, input spectra ring */
    float *time;       /* CAB_FFT_N: previous block | current block */
    float *acc;        /* CAB_FFT_N: accumulated output spectrum */
    float *work;       /* CAB_FFT_N: pffft scratch */
} cab_conv_t;

/* One setup serves every instance - pffft setups are read-only once built. */
static PFFFT_Setup *cab_fft_setup(void) {
    static PFFFT_Setup *setup = pffft_new_setup(CAB_FFT_N, PFFFT_REAL);
    return setup;
}

static float *cab_alloc(size_t n) {
    float *p = (float *)pffft_aligned_malloc(n * sizeof(float));
    if (p) memset(p, 0, n * sizeof(float));
    return p;
}

static void cab_conv_free(cab_conv_t *c) {
    if (!c) return;
    pffft_aligned_free(c->ir_spec);
    pffft_aligned_free(c->fdl);
    pffft_aligned_free(c->time);
    pffft_aligned_free(c->acc);
    pffft_aligned_free(c->work);
    delete c;
}

/* Build a convolver for ir[0..ir_len). Allocates; call off the audio path
 * where possible. Returns nullptr on failure. */
static cab_conv_t *cab_conv_create(const float *ir, int ir_len) {
    PFFFT_Setup *setup = cab_fft_setup();
    if (!setup || !ir || ir_len <= 0) return nullptr;

    cab_conv_t *c = new cab_conv_t();
    c->parts = (ir_len + CAB_BLOCK - 1) / CAB_BLOCK;
    c->fdl_pos = 0;
    c->ir_spec = cab_alloc((size_t)c->parts * CAB_FFT_N);
    c->fdl     = cab_alloc((size_t)c->parts * CAB_FFT_N);
    c->time    = cab_alloc(CAB_FFT_N);
    c->acc     = cab_alloc(CAB_FFT_N);
    c->work    = cab_alloc(CAB_FFT_N);
    if (!c->ir_spec || !c->fdl || !c->time || !c->acc || !c->work) {
        cab_conv_free(c);
        return nullptr;
    }

    /* Partition p = ir[p*B .. p*B+B), zero-padded to 2B. The 1/N the
     * inverse FFT needs is folded in here, once. */
    const float scale = 1.0f / CAB_FFT_N;
    for (int p = 0; p < c->parts; p++) {
        memset(c->time, 0, CAB_FFT_N * sizeof(float));
        int start = p * CAB_BLOCK;
        int len = ir_len - start < CAB_BLOCK ? ir_len - start : CAB_BLOCK;
        for (int i = 0; i < len; i++) c->time[i] = ir[start + i] * scale;
        pffft_transform(setup, c->time, c->ir_spec + (size_t)p * CAB_FFT_N,
                        c->work, PFFFT_FORWARD);
    }
    memset(c->time, 0, CAB_FFT_N * sizeof(float));
    return c;
}

/* Convolve exactly CAB_BLOCK samples in place. */
static void cab_conv_process(cab_conv_t *c, float *audio) {
    PFFFT_Setup *setup = cab_fft_setup();

    /* Slide the input window: [previous block | this block]. */
    memcpy(c->time, c->time + CAB_BLOCK, CAB_BLOCK * sizeof(float));
    memcpy(c->time + CAB_BLOCK, audio, CAB_BLOCK * sizeof(float));

    float *newest = c->fdl + (size_t)c->fdl_pos * CAB_FFT_N;
    pffft_transform(setup, c->time, newest, c->work, PFFFT_FORWARD);

    /* Partition p meets the input from p blocks ago. */
    memset(c->acc, 0, CAB_FFT_N * sizeof(float));
    int slot = c->fdl_pos;
    for (int p = 0; p < c->parts; p++) {
        pffft_zconvolve_accumulate(setup, c->fdl + (size_t)slot * CAB_FFT_N,
                                   c->ir_spec + (size_t)p * CAB_FFT_N,
                                   c->acc, 1.0f);
        if (--slot < 0) slot = c->parts - 1;
    }

    /* Overlap-save: the second half is the valid linear convolution. */
    pffft_transform(setup, c->acc, c->acc, c->work, PFFFT_BACKWARD);
    memcpy(audio, c->acc + CAB_BLOCK, CAB_BLOCK * sizeof(float));

    if (++c->fdl_pos >= c->parts) c->fdl_pos = 0;
}

#endif /* NAM_CAB_CONV_H */
