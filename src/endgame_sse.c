/**
 * @file endgame_avx.c
 *
 *
 * SSE / AVX optimized version of endgame.c for the last four empties.
 *
 * Bitboard and empty list is kept in SSE registers, but performance gain
 * is limited for GCC minGW build since vectorcall is not supported.
 *
 * @date 1998 - 2018
 * @author Richard Delorme
 * @author Toshihiko Okuhara
 * @version 4.4
 * 
 */

#include "settings.h"
#include "search.h"
#include <assert.h>

#define	SWAP64	0x4e	// for _mm_shuffle_epi32
#define	DUPLO	0x44
#define	DUPHI	0xee

#if defined(__AVX__) && (defined(__x86_64__) || defined(_M_X64))
#define	EXTRACT_P(OP)	_mm_extract_epi64(OP, 1)
#else
#define	EXTRACT_P(OP)	_mm_cvtsi128_si64(_mm_shuffle_epi32(OP, DUPHI))
#endif

#ifdef __AVX__
#define TESTZ_FLIP(X)	_mm_testz_si128(X, X)
#else
#define	TESTZ_FLIP(X)	(_mm_cvtsi128_si32(_mm_packs_epi16(X, X)) == 0)
#endif

// in count_last_flip_sse.c
extern const unsigned char COUNT_FLIP[8][256];
extern const V2DI mask_hdvd[64][2];

/**
 * @brief Compute a board resulting of a move played on a previous board.
 *
 * @param OP board to play the move on.
 * @param x move to play.
 * @param next resulting board.
 * @return true if no flips.
 */
static inline int vectorcall board_next_sse(__m128i OP, int x, __m128i *next)
{
	__m128i flipped = mm_Flip(OP, x);
	OP = _mm_xor_si128(OP, _mm_or_si128(flipped, _mm_loadl_epi64((__m128i *) &X_TO_BIT[x])));
	*next = _mm_shuffle_epi32(OP, SWAP64);
	return TESTZ_FLIP(flipped);
}

/**
 * @brief Get the final score.
 *
 * Get the final score, when no move can be made.
 *
 * @param OP Board.
 * @param n_empties Number of empty squares remaining on the board.
 * @return The final score, as a disc difference.
 */
static int vectorcall board_solve_sse(__m128i OP, int n_empties)
{
	int score = bit_count(_mm_cvtsi128_si64(OP)) * 2 - SCORE_MAX;	// in case of opponents win
	int diff = score + n_empties;		// = n_discs_p - (64 - n_empties - n_discs_p)

	SEARCH_STATS(++statistics.n_search_solve);

	if (diff >= 0)
		score = diff;
	if (diff > 0)
		score += n_empties;
	return score;
}

/**
 * @brief Get the final score.
 *
 * Get the final score, when 1 empty squares remain.
 * The following code has been adapted from Zebra by Gunnar Anderson.
 *
 * @param OP  Board to evaluate.
 * @param beta   Beta bound.
 * @param pos    Last empty square to play.
 * @return       The final opponent score, as a disc difference.
 */
static int vectorcall board_score_sse_1(__m128i OP, const int beta, const int pos)
{
	int	score, score2, n_flips, i;
	const unsigned char *COUNT_FLIP_X = COUNT_FLIP[pos & 7];
	const unsigned char *COUNT_FLIP_Y = COUNT_FLIP[pos >> 3];
	__m128i	PP, II, OO;
	__m128i M_hd = mask_hdvd[pos][0].v2;
	__m128i M_vd = mask_hdvd[pos][1].v2;

	score = 2 * bit_count(EXTRACT_P(OP)) - SCORE_MAX;

	// n_flips = last_flip(x, board->player);
	PP = _mm_shuffle_epi32(OP, DUPLO);
	II = _mm_sad_epu8(_mm_and_si128(PP, M_hd), _mm_setzero_si128());
	n_flips  = COUNT_FLIP_X[_mm_cvtsi128_si32(II)];
	n_flips += COUNT_FLIP_X[_mm_extract_epi16(II, 4)];
	i = _mm_movemask_epi8(_mm_sub_epi8(_mm_setzero_si128(), _mm_and_si128(PP, M_vd)));
	n_flips += COUNT_FLIP_Y[i >> 8];
	n_flips += COUNT_FLIP_Y[(unsigned char) i];

	if (n_flips != 0) {
		score -= n_flips;

	} else {
		score2 = score + 2;	// empty for player
		if (score >= 0)
			score = score2;

		if (score < beta) {	// lazy cut-off
			// n_flips = last_flip(x, board->opponent);
			OO = _mm_shuffle_epi32(OP, DUPHI);
			II = _mm_sad_epu8(_mm_and_si128(OO, M_hd), _mm_setzero_si128());
			n_flips  = COUNT_FLIP_X[_mm_cvtsi128_si32(II)];
			n_flips += COUNT_FLIP_X[_mm_extract_epi16(II, 4)];
			i = _mm_movemask_epi8(_mm_sub_epi8(_mm_setzero_si128(), _mm_and_si128(OO, M_vd)));
			n_flips += COUNT_FLIP_Y[i >> 8];
			n_flips += COUNT_FLIP_Y[(unsigned char) i];

			if (n_flips != 0) {
				score = score2 + n_flips;
			}
		}
	}

	return score;
}

/**
 * @brief Get the final score.
 *
 * Get the final score, when 2 empty squares remain.
 *
 * @param OP The board to evaluate.
 * @param empties Packed empty square coordinates.
 * @param alpha Alpha bound.
 * @param n_nodes Node counter.
 * @return The final score, as a disc difference.
 */
static int vectorcall board_solve_sse_2(__m128i OP, __m128i empties, int alpha, volatile unsigned long long *n_nodes)
{
	__m128i next, PO;
	int score, bestscore, nodes;
	int x1 = _mm_extract_epi16(empties, 1);
	int x2 = _mm_extract_epi16(empties, 0);
	unsigned long long bb;
	// const int beta = alpha + 1;

	SEARCH_STATS(++statistics.n_board_solve_2);
	nodes = 0;
	SEARCH_UPDATE_INTERNAL_NODES(nodes);

	bestscore = -SCORE_INF;
	bb = EXTRACT_P(OP);	// opponent
	if ((NEIGHBOUR[x1] & bb) && !board_next_sse(OP, x1, &next)) {
		SEARCH_UPDATE_INTERNAL_NODES(nodes);
		bestscore = board_score_sse_1(next, alpha + 1, x2);
	}

	if (bestscore <= alpha) {
		if ((NEIGHBOUR[x2] & bb) && !board_next_sse(OP, x2, &next)) {
			SEARCH_UPDATE_INTERNAL_NODES(nodes);
			score = board_score_sse_1(next, alpha + 1, x1);
			if (score > bestscore) bestscore = score;
		}

		// pass
		if (bestscore == -SCORE_INF) {
			bestscore = SCORE_INF;
			bb = _mm_cvtsi128_si64(OP);	// player
			PO = _mm_shuffle_epi32(OP, SWAP64);
			if ((NEIGHBOUR[x1] & bb) && !board_next_sse(PO, x1, &next)) {
				SEARCH_UPDATE_INTERNAL_NODES(nodes);
				bestscore = -board_score_sse_1(next, -alpha, x2);
			}

			if (bestscore > alpha) {
				if ((NEIGHBOUR[x2] & bb) && !board_next_sse(PO, x2, &next)) {
					SEARCH_UPDATE_INTERNAL_NODES(nodes);
					score = -board_score_sse_1(next, -alpha, x1);
					if (score < bestscore) bestscore = score;
				}
				// gameover
				if (bestscore == SCORE_INF) bestscore = board_solve_sse(OP, 2);
			}
		}
	}

	*n_nodes += nodes;
 	assert(SCORE_MIN <= bestscore && bestscore <= SCORE_MAX);
 	assert((bestscore & 1) == 0);
	return bestscore;
}

/**
 * @brief Get the final score.
 *
 * Get the final score, when 3 empty squares remain.
 *
 * @param OP The board to evaluate.
 * @param empties Packed empty square coordinates.
 * @param alpha Alpha bound.
 * @param parity Parity flags.
 * @param n_nodes Node counter.
 * @return The final score, as a disc difference.
 */
static int vectorcall search_solve_sse_3(__m128i OP, __m128i empties, int alpha, unsigned int parity, volatile unsigned long long *n_nodes)
{
	__m128i next, PO;
	int score, bestscore, x;
	unsigned long long bb;
	// const int beta = alpha + 1;

	SEARCH_STATS(++statistics.n_search_solve_3);
	SEARCH_UPDATE_INTERNAL_NODES(*n_nodes);

	empties = _mm_unpacklo_epi8(empties, _mm_setzero_si128());	// to ease shuffle
	// parity based move sorting
	if (!(parity & QUADRANT_ID[_mm_extract_epi16(empties, 2)])) {
		if (parity & QUADRANT_ID[_mm_extract_epi16(empties, 1)]) {
			empties = _mm_shufflelo_epi16(empties, 0xd8); // case 1(x2) 2(x1 x3)
		} else {
			empties = _mm_shufflelo_epi16(empties, 0xc9); // case 1(x3) 2(x1 x2)
		}
	}

	// best move alphabeta search
	bestscore = -SCORE_INF;
	bb = EXTRACT_P(OP);	// opponent
	x = _mm_extract_epi16(empties, 2);
	if ((NEIGHBOUR[x] & bb) && !board_next_sse(OP, x, &next)) {
		bestscore = -board_solve_sse_2(next, empties, -(alpha + 1), n_nodes);
		if (bestscore > alpha) return bestscore;
	}

	x = _mm_extract_epi16(empties, 1);
	if ((NEIGHBOUR[x] & bb) && !board_next_sse(OP, x, &next)) {
		score = -board_solve_sse_2(next, _mm_shufflelo_epi16(empties, 0xd8), -(alpha + 1), n_nodes);
		if (score > alpha) return score;
		else if (score > bestscore) bestscore = score;
	}

	x = _mm_extract_epi16(empties, 0);
	if ((NEIGHBOUR[x] & bb) && !board_next_sse(OP, x, &next)) {
		score = -board_solve_sse_2(next, _mm_shufflelo_epi16(empties, 0xc9), -(alpha + 1), n_nodes);
		if (score > bestscore) bestscore = score;
	}

	// pass ?
	if (bestscore == -SCORE_INF) {
		// best move alphabeta search
		bestscore = SCORE_INF;
		bb = _mm_cvtsi128_si64(OP);	// player
		PO = _mm_shuffle_epi32(OP, SWAP64);
		x = _mm_extract_epi16(empties, 2);
		if ((NEIGHBOUR[x] & bb) && !board_next_sse(PO, x, &next)) {
			bestscore = board_solve_sse_2(next, empties, alpha, n_nodes);
			if (bestscore <= alpha) return bestscore;
		}

		x = _mm_extract_epi16(empties, 1);
		if ((NEIGHBOUR[x] & bb) && !board_next_sse(PO, x, &next)) {
			score = board_solve_sse_2(next, _mm_shufflelo_epi16(empties, 0xd8), alpha, n_nodes);
			if (score <= alpha) return score;
			else if (score < bestscore) bestscore = score;
		}

		x = _mm_extract_epi16(empties, 0);
		if ((NEIGHBOUR[x] & bb) && !board_next_sse(PO, x, &next)) {
			score = board_solve_sse_2(next, _mm_shufflelo_epi16(empties, 0xc9), alpha, n_nodes);
			if (score < bestscore) bestscore = score;
		}

		// gameover
		if (bestscore == SCORE_INF) bestscore = board_solve_sse(OP, 3);
	}

 	assert(SCORE_MIN <= bestscore && bestscore <= SCORE_MAX);
	return bestscore;
}

/**
 * @brief Get the final score.
 *
 * Get the final score, when 4 empty squares remain.
 *
 * @param search Search position.
 * @param alpha Upper score value.
 * @return The final score, as a disc difference.
 */
int search_solve_4(Search *search, const int alpha)
{
	__m128i	OP, next, empties_series;
	SquareList *empty;
	int x1, x2, x3, x4;
	int score, bestscore;
	unsigned int parity;
	unsigned long long opp;
	// const int beta = alpha + 1;

	SEARCH_STATS(++statistics.n_search_solve_4);
	SEARCH_UPDATE_INTERNAL_NODES(search->n_nodes);

	// stability cutoff
	if (search_SC_NWS(search, alpha, &score)) return score;

	OP = _mm_loadu_si128((__m128i *) search->board);
	x1 = (empty = search->empties->next)->x;
	x2 = (empty = empty->next)->x;
	x3 = (empty = empty->next)->x;
	x4 = empty->next->x;
#ifdef __AVX__
	empties_series = _mm_cvtsi32_si128((x4 << 24) | (x3 << 16) | (x2 << 8) | x1);
	empties_series = _mm_shuffle_epi8(empties_series, _mm_set_epi8(3, 0, 1, 2, 2, 0, 1, 3, 1, 0, 2, 3, 0, 1, 2, 3));
#else
	empties_series = _mm_set_epi8(x4, x1, x2, x3, x3, x1, x2, x4, x2, x1, x3, x4, x1, x2, x3, x4);
#endif

	// parity based move sorting.
	// The following hole sizes are possible:
	//    4 - 1 3 - 2 2 - 1 1 2 - 1 1 1 1
	// Only the 1 1 2 case needs move sorting.
	parity = search->parity;
	if (!(parity & QUADRANT_ID[x1])) {
		if (parity & QUADRANT_ID[x2]) {
			if (parity & QUADRANT_ID[x3]) { // case 1(x2) 1(x3) 2(x1 x4)
				empties_series = _mm_shuffle_epi32(empties_series, 0xc9);
			} else { // case 1(x2) 1(x4) 2(x1 x3)
				empties_series = _mm_shuffle_epi32(empties_series, 0x8d);
			}
		} else if (parity & QUADRANT_ID[x3]) { // case 1(x3) 1(x4) 2(x1 x2)
			empties_series = _mm_shuffle_epi32(empties_series, 0x4e);
		}
	} else {
		if (!(parity & QUADRANT_ID[x2])) {
			if (parity & QUADRANT_ID[x3]) { // case 1(x1) 1(x3) 2(x2 x4)
				empties_series = _mm_shuffle_epi32(empties_series, 0xd8);
			} else { // case 1(x1) 1(x4) 2(x2 x3)
				empties_series = _mm_shuffle_epi32(empties_series, 0x9c);
			}
		}
	}

	// best move alphabeta search
	bestscore = -SCORE_INF;
	opp = EXTRACT_P(OP);
	x1 = _mm_cvtsi128_si32(empties_series) >> 24;
	if ((NEIGHBOUR[x1] & opp) && !board_next_sse(OP, x1, &next)) {
		bestscore = -search_solve_sse_3(next, empties_series, -(alpha + 1), parity ^ QUADRANT_ID[x1], &search->n_nodes);
		if (bestscore > alpha) return bestscore;
	}

	empties_series = _mm_shuffle_epi32(empties_series, 0x39);
	x2 = _mm_cvtsi128_si32(empties_series) >> 24;
	if ((NEIGHBOUR[x2] & opp) && !board_next_sse(OP, x2, &next)) {
		score = -search_solve_sse_3(next, empties_series, -(alpha + 1), parity ^ QUADRANT_ID[x2], &search->n_nodes);
		if (score > alpha) return score;
		else if (score > bestscore) bestscore = score;
	}

	empties_series = _mm_shuffle_epi32(empties_series, 0x39);
	x3 = _mm_cvtsi128_si32(empties_series) >> 24;
	if ((NEIGHBOUR[x3] & opp) && !board_next_sse(OP, x3, &next)) {
		score = -search_solve_sse_3(next, empties_series, -(alpha + 1), parity ^ QUADRANT_ID[x3], &search->n_nodes);
		if (score > alpha) return score;
		else if (score > bestscore) bestscore = score;
	}

	empties_series = _mm_shuffle_epi32(empties_series, 0x39);
	x4 = _mm_cvtsi128_si32(empties_series) >> 24;
	if ((NEIGHBOUR[x4] & opp) && !board_next_sse(OP, x4, &next)) {
		score = -search_solve_sse_3(next, empties_series, -(alpha + 1), parity ^ QUADRANT_ID[x4], &search->n_nodes);
		if (score > bestscore) bestscore = score;
	}

	// no move
	if (bestscore == -SCORE_INF) {
		if (can_move(opp, _mm_cvtsi128_si64(OP))) { // pass
			search_pass_endgame(search);
			bestscore = -search_solve_4(search, -(alpha + 1));
			search_pass_endgame(search);
		} else { // gameover
			bestscore = board_solve_sse(OP, 4);
		}
	}

 	assert(SCORE_MIN <= bestscore && bestscore <= SCORE_MAX);
	return bestscore;
}
