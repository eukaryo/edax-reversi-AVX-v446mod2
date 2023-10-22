/**
 * @file eval.h
 *
 * Evaluation function's header.
 *
 * @date 1998 - 2018
 * @author Richard Delorme
 * @version 4.4
 */

#ifndef EDAX_EVAL_H
#define EDAX_EVAL_H

#include "bit.h"

/** number of features */
enum { EVAL_N_FEATURE = 47 };

/**
 * struct Eval
 * @brief evaluation function
 */
typedef struct Eval {
	union {
		unsigned short us[EVAL_N_FEATURE];         /**!< discs' features */
#if defined(hasSSE2) || defined(USE_MSVC_X86)
		__m128i	v8[6];
#endif
#ifdef __AVX2__
		__m256i	v16[3];
#endif
	} feature;
	int player;
} Eval;

struct Board;
struct Move;

/** number of (unpacked) weights */
enum { EVAL_N_WEIGHT = 226315 };

/** number of plies */
enum { EVAL_N_PLY = 61 };

extern short (*EVAL_WEIGHT)[EVAL_N_PLY][EVAL_N_WEIGHT];


/* function declaration */
void eval_open(const char*);
void eval_close(void);
// void eval_init(Eval*);
// void eval_free(Eval*);
void eval_set(Eval*, const struct Board*);
void eval_update(Eval*, const struct Move*);
void eval_restore(Eval*, const struct Move*);
void eval_pass(Eval*);
double eval_sigma(const int, const int, const int);

#endif

