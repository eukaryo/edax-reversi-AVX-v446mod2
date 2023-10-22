/**
 * @file board.h
 *
 * Board management header file.
 *
 * @date 1998 - 2017
 * @author Richard Delorme
 * @version 4.4
 */

#ifndef EDAX_BOARD_H
#define EDAX_BOARD_H

#include "const.h"
#include "settings.h"
#include "bit.h"

#include <stdio.h>
#include <stdbool.h>

/** Board : board representation */
typedef struct Board {
	unsigned long long player, opponent;     /**< bitboard representation */
} Board;

struct Move;
struct Random;

/* function declarations */
void board_init(Board*);
int board_set(Board*, const char*);
int board_from_FEN(Board*, const char*);
int board_compare(const Board*, const Board*);
bool board_equal(const Board*, const Board*);
void board_symetry(const Board*, const int, Board*);
int board_unique(const Board*, Board*);
void board_check(const Board*);
void board_rand(Board*, int, struct Random*);

int board_count_last_flips(const Board*, const int);
unsigned long long board_get_move(const Board*, const int, struct Move*);
bool board_check_move(const Board*, struct Move*);
void board_swap_players(Board*);
void board_update(Board*, const struct Move*);
void board_restore(Board*, const struct Move*);
void board_pass(Board*);
unsigned long long board_next(const Board*, const int, Board*);
unsigned long long board_pass_next(const Board*, const int, Board*);
unsigned long long board_get_hash_code(const Board*);
int board_get_square_color(const Board*, const int);
bool board_is_occupied(const Board*, const int);
void board_print(const Board*, const int, FILE*);
char* board_to_string(const Board*, const int, char *);
void board_print_FEN(const Board*, const int, FILE*);
char* board_to_FEN(const Board*, const int, char*);
bool board_is_pass(const Board*);
bool board_is_game_over(const Board*);
int board_count_empties(const Board *board);

unsigned long long get_moves(const unsigned long long, const unsigned long long);
bool can_move(const unsigned long long, const unsigned long long);
unsigned long long get_moves_6x6(const unsigned long long, const unsigned long long);
bool can_move_6x6(const unsigned long long, const unsigned long long);
int get_mobility(const unsigned long long, const unsigned long long);
int get_weighted_mobility(const unsigned long long, const unsigned long long);
int get_potential_mobility(const unsigned long long, const unsigned long long);
void edge_stability_init(void);
int get_stability(const unsigned long long, const unsigned long long);
int get_edge_stability(const unsigned long long, const unsigned long long);
int get_corner_stability(const unsigned long long);

#if defined(USE_GAS_MMX) || defined(USE_MSVC_X86)
void init_mmx (void);
unsigned long long get_moves_mmx(unsigned int PL, unsigned int PH, unsigned int OL, unsigned int OH);
unsigned long long get_moves_sse(unsigned int PL, unsigned int PH, unsigned int OL, unsigned int OH);
int get_stability_mmx(unsigned int PL, unsigned int PH, unsigned int OL, unsigned int OH);
int get_potential_mobility_mmx(unsigned long long P, unsigned long long O);
#endif
#if defined(USE_GAS_MMX) && defined(__3dNOW__)
unsigned long long board_get_hash_code_mmx(const unsigned char *p);
#elif defined(USE_GAS_MMX) || defined(USE_MSVC_X86)
unsigned long long board_get_hash_code_sse(const unsigned char *p);
#endif

extern unsigned char edge_stability[256 * 256];
extern const unsigned long long A1_A8[256];

#if ((LAST_FLIP_COUNTER == COUNT_LAST_FLIP_PLAIN) || (LAST_FLIP_COUNTER == COUNT_LAST_FLIP_SSE) || (LAST_FLIP_COUNTER == COUNT_LAST_FLIP_BMI2))
	extern int last_flip(int pos, unsigned long long P);
#else
	extern int (*count_last_flip[BOARD_SIZE + 1])(const unsigned long long);
	#define	last_flip(x,P)	count_last_flip[x](P)
#endif

#if MOVE_GENERATOR == MOVE_GENERATOR_AVX
	extern __m128i vectorcall mm_Flip(const __m128i OP, int pos);
	#define	Flip(x,P,O)	((unsigned long long) _mm_cvtsi128_si64(mm_Flip(_mm_unpacklo_epi64(_mm_cvtsi64_si128(P), _mm_cvtsi64_si128(O)), (x))))
	#define	board_flip(board,x)	((unsigned long long) _mm_cvtsi128_si64(mm_Flip(_mm_loadu_si128((__m128i *) (board)), (x))))

#elif MOVE_GENERATOR == MOVE_GENERATOR_SSE
	extern __m128i (vectorcall *mm_flip[])(const __m128i);
	#define	Flip(x,P,O)	((unsigned long long) _mm_cvtsi128_si64(mm_flip[x](_mm_unpacklo_epi64(_mm_cvtsi64_si128(P), _mm_cvtsi64_si128(O)))))
	#define mm_Flip(OP,x)	mm_flip[x](OP)
	#define	board_flip(board,x)	((unsigned long long) _mm_cvtsi128_si64(mm_flip[x](_mm_loadu_si128((__m128i *) (board)))))

#elif MOVE_GENERATOR == MOVE_GENERATOR_SSE_BSWAP
	extern unsigned long long flip(int, const unsigned long long, const unsigned long long);
	#define	Flip(x,P,O)	flip((x), (P), (O))
	#define	board_flip(board,x)	flip((x), (board)->player, (board)->opponent)

#elif MOVE_GENERATOR == MOVE_GENERATOR_32
	extern unsigned long long (*flip[BOARD_SIZE + 2])(unsigned int, unsigned int, unsigned int, unsigned int);
	#define Flip(x,P,O)	flip[x]((unsigned int)(P), (unsigned int)((P) >> 32), (unsigned int)(O), (unsigned int)((O) >> 32))
	#define	board_flip(board,x)	flip[x]((unsigned int)((board)->player), ((unsigned int *) &board->player)[1], (unsigned int)((board)->opponent), ((unsigned int *) &board->opponent)[1])
	#if defined(USE_GAS_MMX) && !defined(hasSSE2)
		extern void init_flip_sse(void);
	#endif

#else
	extern unsigned long long (*flip[BOARD_SIZE + 2])(const unsigned long long, const unsigned long long);
	#define	Flip(x,P,O)	flip[x]((P), (O))
	#define	board_flip(board,x)	flip[x]((board)->player, (board)->opponent)
#endif

#endif
