/**
 * @file settings.h
 *
 * Various macro / constants to control algorithm usage.
 *
 * @date 1998 - 2018
 * @author Richard Delorme
 * @version 4.4
 */


#ifndef EDAX_SETTINGS_H
#define EDAX_SETTINGS_H

#include <stdbool.h>

#define MOVE_GENERATOR_CARRY 1		// 32.6Mnps
#define MOVE_GENERATOR_KINDERGARTEN 2	// 31.1Mnps
#define MOVE_GENERATOR_SSE 3		// 34.4Mnps
#define MOVE_GENERATOR_BITSCAN 4	// 32.7Mnps
#define MOVE_GENERATOR_ROXANE 5		// 29.0Mnps
#define MOVE_GENERATOR_32 6		// 31.3Mnps
#define MOVE_GENERATOR_SSE_BSWAP 7	// 30.6Mnps
#define MOVE_GENERATOR_AVX 8		// 34.7Mnps

#define	COUNT_LAST_FLIP_CARRY 1		// 33.8Mnps
#define COUNT_LAST_FLIP_KINDERGARTEN 2	// 33.5Mnps
#define COUNT_LAST_FLIP_SSE 3		// 34.7Mnps
#define COUNT_LAST_FLIP_BITSCAN 4	// 33.9Mnps
#define COUNT_LAST_FLIP_PLAIN 5		// 33.3Mnps
#define COUNT_LAST_FLIP_32 6		// 33.1Mnps
#define COUNT_LAST_FLIP_BMI2 7		// 34.7Mnps	// slow on AMD

/**move generation. */
#ifndef MOVE_GENERATOR
	#ifdef __AVX2__
		#define MOVE_GENERATOR MOVE_GENERATOR_AVX
	#elif defined(hasSSE2)
		#define MOVE_GENERATOR MOVE_GENERATOR_SSE
	#else
		#define MOVE_GENERATOR MOVE_GENERATOR_32
	#endif
#endif
#ifndef LAST_FLIP_COUNTER
	#ifdef hasSSE2
		#define LAST_FLIP_COUNTER COUNT_LAST_FLIP_SSE
	#else
		#define LAST_FLIP_COUNTER COUNT_LAST_FLIP_32
	#endif
#endif

/** transposition cutoff usage. */
#define USE_TC true

/** stability cutoff usage. */
#define USE_SC true

/** enhanced transposition cutoff usage. */
#define USE_ETC true

/** probcut usage. */
#define USE_PROBCUT true

/** Use recursive probcut */
#define USE_RECURSIVE_PROBCUT true

/** limit recursive probcut level */
#define LIMIT_RECURSIVE_PROBCUT(x) x

/** kogge-stone parallel prefix algorithm usage.
 *  0 -> none, 1 -> move generator, 2 -> stability, 3 -> both.
 */
#define KOGGE_STONE 2

/** 1 stage parallel prefix algorithm usage.
 *  0 -> none, 1 -> move generator, 2 -> stability, 3 -> both.
 */
#define PARALLEL_PREFIX 1

#if (KOGGE_STONE & PARALLEL_PREFIX)
	#error "usage of 2 incompatible algorithms"
#endif

/** Internal Iterative Deepening. */
#define USE_IID false

/** Use previous search result */
#define USE_PREVIOUS_SEARCH true

/** Allow type puning */
#ifndef USE_TYPE_PUNING
#ifdef ANDROID
#define USE_TYPE_PUNING false
#else
#define USE_TYPE_PUNING true
#endif
#endif

/** Hash-n-way. */
#define HASH_N_WAY 4

/** hash align */
#define HASH_ALIGNED 1

/** PV extension (solve PV alone sooner) */
#define USE_PV_EXTENSION true

/** Swith from endgame to shallow search (faster but less node efficient) at this depth. */
#define DEPTH_TO_SHALLOW_SEARCH 7

/** Switch from midgame to endgame search (faster but less node efficient) at this depth. */
#define DEPTH_MIDGAME_TO_ENDGAME 15

/** Switch from midgame result (evaluated score) to endgame result (exact score) at this number of empties. */
#define ITERATIVE_MIN_EMPTIES 10

/** Store bestmoves in the pv_hash up to this height. */
#define PV_HASH_HEIGHT 5

/** Try ETC down to this depth. */
#define ETC_MIN_DEPTH 5

/** bound for usefull move sorting */
#define SORT_ALPHA_DELTA 8

/** Try Node splitting (for parallel search) down to that depth. */
#define SPLIT_MIN_DEPTH 5

/** Stop Node splitting (for parallel search) when few move remains.  */
#define SPLIT_MIN_MOVES_TODO 1

/** Stop Node splitting (for parallel search) after a few splitting.  */
#define SPLIT_MAX_SLAVES 3

/** Branching factor (to adjust alloted time). */
#define BRANCHING_FACTOR 2.24

/** Parallelisable work. */
#define SMP_W 49.0

/** Critical time. */
#define SMP_C 1.0

/** Fast perft */
#define  FAST_PERFT true

/** multi_pv depth */
#define MULTIPV_DEPTH 10

#endif /* EDAX_SETTINGS_H */

