#pragma once

#ifndef DENOISE_H
#define DENOISE_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "rnnoise.h"
#include "pitch.h"
#include "arch.h"
#include "rnn.h"
#include "rnn_data.h"

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)

#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define SQUARE(x) ((x)*(x))

#define NB_BANDS 22

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)

#define M_PI  3.141592653


#ifndef TRAINING
#define TRAINING 0
#endif




typedef struct {
	int init;
	kiss_fft_state* kfft;
	float half_window[FRAME_SIZE];
	float dct_table[NB_BANDS * NB_BANDS];
} CommonState;

struct DenoiseState {
	float analysis_mem[FRAME_SIZE];
	float cepstral_mem[CEPS_MEM][NB_BANDS];
	int memid;
	float synthesis_mem[FRAME_SIZE];
	float pitch_buf[PITCH_BUF_SIZE];
	float pitch_enh_buf[PITCH_BUF_SIZE];
	float last_gain;
	int last_period;
	float mem_hp_x[2];
	float lastg[NB_BANDS];
	RNNState rnn;
};


void compute_band_energy(float* bandE, const kiss_fft_cpx* X);

void compute_band_corr(float* bandE, const kiss_fft_cpx* X, const kiss_fft_cpx* P);

void interp_band_gain(float* g, const float* bandE);

void pitch_filter(kiss_fft_cpx* X, const kiss_fft_cpx* P, const float* Ex, const float* Ep,
	const float* Exp, const float* g);


#endif