/**
 * @file endgame.c
 *
 * Search near the end of the game.
 *
 * @date 1998 - 2017
 * @author Richard Delorme
 * @version 4.4
 */


#include "search.h"

#include "bit.h"
#include "settings.h"
#include "stats.h"
#include "ybwc.h"

#include <assert.h>

#if LAST_FLIP_COUNTER == COUNT_LAST_FLIP_CARRY
	#include "count_last_flip_carry_64.c"
#elif LAST_FLIP_COUNTER == COUNT_LAST_FLIP_SSE
	#include "count_last_flip_sse.c"
#elif LAST_FLIP_COUNTER == COUNT_LAST_FLIP_BITSCAN
	#include "count_last_flip_bitscan.c"
#elif LAST_FLIP_COUNTER == COUNT_LAST_FLIP_PLAIN
	#include "count_last_flip_plain.c"
#elif LAST_FLIP_COUNTER == COUNT_LAST_FLIP_32
	#include "count_last_flip_32.c"
#elif LAST_FLIP_COUNTER == COUNT_LAST_FLIP_BMI2
	#include "count_last_flip_bmi2.c"
#else // LAST_FLIP_COUNTER == COUNT_LAST_FLIP_KINDERGARTEN
	#include "count_last_flip_kindergarten.c"
#endif

#if ((MOVE_GENERATOR == MOVE_GENERATOR_AVX) || (MOVE_GENERATOR == MOVE_GENERATOR_SSE)) && (LAST_FLIP_COUNTER == COUNT_LAST_FLIP_SSE)
	#include "endgame_sse.c"	// vectorcall version
#endif

/**
 * @brief Get the final score.
 *
 * Get the final score, when no move can be made.
 *
 * @param board Board.
 * @param n_empties Number of empty squares remaining on the board.
 * @return The final score, as a disc difference.
 */
static int board_solve(const Board *board, const int n_empties)
{
	int score = bit_count(board->player) * 2 - SCORE_MAX;	// in case of opponents win
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
 * Get the final score, when no move can be made.
 *
 * @param search Search.
 * @return The final score, as a disc difference.
 */
int search_solve(const Search *search)
{
	return board_solve(search->board, search->n_empties);
}

/**
 * @brief Get the final score.
 *
 * Get the final score, when the board is full.
 *
 * @param search Search.
 * @return The final score, as a disc difference.
 */
int search_solve_0(const Search *search)
{
	SEARCH_STATS(++statistics.n_search_solve_0);

	return 2 * bit_count(search->board->player) - SCORE_MAX;
}

/**
 * @brief Get the final score.
 *
 * Get the final score, when 1 empty squares remain.
 * The following code has been adapted from Zebra by Gunnar Anderson.
 *
 * @param board  Board to evaluate.
 * @param beta   Beta bound.
 * @param x      Last empty square to play.
 * @return       The final opponent score, as a disc difference.
 */
int board_score_1(const Board *board, const int beta, const int x)
{
	int score, score2, n_flips;

	score = 2 * bit_count(board->opponent) - SCORE_MAX;

	if ((n_flips = last_flip(x, board->player)) != 0) {
		score -= n_flips;
	} else {
		score2 = score + 2;	// empty for player
		if (score >= 0)
			score = score2;
		if (score < beta) {	// lazy cut-off
			if ((n_flips = last_flip(x, board->opponent)) != 0) {
				score = score2 + n_flips;
			}
		}
	}

	return score;
}

#if !(((MOVE_GENERATOR == MOVE_GENERATOR_AVX) || (MOVE_GENERATOR == MOVE_GENERATOR_SSE)) && (LAST_FLIP_COUNTER == COUNT_LAST_FLIP_SSE))
/**
 * @brief Get the final score.
 *
 * Get the final score, when 2 empty squares remain.
 *
 * @param board The board to evaluate.
 * @param alpha Alpha bound.
 * @param x1 First empty square coordinate.
 * @param x2 Second empty square coordinate.
 * @param search Search.
 * @return The final score, as a disc difference.
 */
static int board_solve_2(Board *board, int alpha, const int x1, const int x2, volatile unsigned long long *n_nodes)
{
	Board next[1];
	int score, bestscore, nodes;
	// const int beta = alpha + 1;

	SEARCH_STATS(++statistics.n_board_solve_2);
	nodes = 0;
	SEARCH_UPDATE_INTERNAL_NODES(nodes);

	bestscore = -SCORE_INF;
	if ((NEIGHBOUR[x1] & board->opponent) && board_next(board, x1, next)) {
		SEARCH_UPDATE_INTERNAL_NODES(nodes);
		bestscore = board_score_1(next, alpha + 1, x2);
	}

	if (bestscore <= alpha) {
		if ((NEIGHBOUR[x2] & board->opponent) && board_next(board, x2, next)) {
			SEARCH_UPDATE_INTERNAL_NODES(nodes);
			score = board_score_1(next, alpha + 1, x1);
			if (score > bestscore) bestscore = score;
		}

		// pass
		if (bestscore == -SCORE_INF) {
			bestscore = SCORE_INF;
			if ((NEIGHBOUR[x1] & board->player) && board_pass_next(board, x1, next)) {
				SEARCH_UPDATE_INTERNAL_NODES(nodes);
				bestscore = -board_score_1(next, -alpha, x2);
			}

			if (bestscore > alpha) {
				if ((NEIGHBOUR[x2] & board->player) && board_pass_next(board, x2, next)) {
					SEARCH_UPDATE_INTERNAL_NODES(nodes);
					score = -board_score_1(next, -alpha, x1);
					if (score < bestscore) bestscore = score;
				}
				// gameover
				if (bestscore == SCORE_INF) bestscore = board_solve(board, 2);
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
 * @param search Search.
 * @param alpha Alpha bound.
 * @return The final score, as a disc difference.
 */
static int search_solve_3(Search *search, const int alpha, Board *board, unsigned int parity)
{
	Board next[1];
	SquareList *empty = search->empties->next;
	int x1 = empty->x;
	int x2 = (empty = empty->next)->x;
	int x3 = empty->next->x;
	int score, bestscore;
	// const int beta = alpha + 1;

	SEARCH_STATS(++statistics.n_search_solve_3);
	SEARCH_UPDATE_INTERNAL_NODES(search->n_nodes);

	// parity based move sorting
	if (!(parity & QUADRANT_ID[x1])) {
		if (parity & QUADRANT_ID[x2]) { // case 1(x2) 2(x1 x3)
			int tmp = x1; x1 = x2; x2 = tmp;
		} else { // case 1(x3) 2(x1 x2)
			int tmp = x1; x1 = x3; x3 = x2; x2 = tmp;
		}
	}

	// best move alphabeta search
	bestscore = -SCORE_INF;
	if ((NEIGHBOUR[x1] & board->opponent) && board_next(board, x1, next)) {
		bestscore = -board_solve_2(next, -(alpha + 1), x2, x3, &search->n_nodes);
		if (bestscore > alpha) return bestscore;
	}

	if ((NEIGHBOUR[x2] & board->opponent) && board_next(board, x2, next)) {
		score = -board_solve_2(next, -(alpha + 1), x1, x3, &search->n_nodes);
		if (score > alpha) return score;
		else if (score > bestscore) bestscore = score;
	}

	if ((NEIGHBOUR[x3] & board->opponent) && board_next(board, x3, next)) {
		score = -board_solve_2(next, -(alpha + 1), x1, x2, &search->n_nodes);
		if (score > bestscore) bestscore = score;
	}

	// pass ?
	if (bestscore == -SCORE_INF) {
		// best move alphabeta search
		bestscore = SCORE_INF;
		if ((NEIGHBOUR[x1] & board->player) && board_pass_next(board, x1, next)) {
			bestscore = board_solve_2(next, alpha, x2, x3, &search->n_nodes);
			if (bestscore <= alpha) return bestscore;
		}

		if ((NEIGHBOUR[x2] & board->player) && board_pass_next(board, x2, next)) {
			score = board_solve_2(next, alpha, x1, x3, &search->n_nodes);
			if (score <= alpha) return score;
			else if (score < bestscore) bestscore = score;
		}

		if ((NEIGHBOUR[x3] & board->player) && board_pass_next(board, x3, next)) {
			score = board_solve_2(next, alpha, x1, x2, &search->n_nodes);
			if (score < bestscore) bestscore = score;
		}

		// gameover
		if (bestscore == SCORE_INF) bestscore = board_solve(board, 3);
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
static int search_solve_4(Search *search, const int alpha)
{
	Board *board;
	Board next[1];
	SquareList *empty;
	int x1, x2, x3, x4;
	int score, bestscore;
	unsigned int parity;
	// const int beta = alpha + 1;

	SEARCH_STATS(++statistics.n_search_solve_4);
	SEARCH_UPDATE_INTERNAL_NODES(search->n_nodes);

	// stability cutoff
	if (search_SC_NWS(search, alpha, &score)) return score;

	board = search->board;
	x1 = (empty = search->empties->next)->x;
	x2 = (empty = empty->next)->x;
	x3 = (empty = empty->next)->x;
	x4 = empty->next->x;

	// parity based move sorting.
	// The following hole sizes are possible:
	//    4 - 1 3 - 2 2 - 1 1 2 - 1 1 1 1
	// Only the 1 1 2 case needs move sorting.
	parity = search->parity;
	if (!(parity & QUADRANT_ID[x1])) {
		if (parity & QUADRANT_ID[x2]) {
			if (parity & QUADRANT_ID[x3]) { // case 1(x2) 1(x3) 2(x1 x4)
				int tmp = x1; x1 = x2; x2 = x3; x3 = tmp;
			} else { // case 1(x2) 1(x4) 2(x1 x3)
				int tmp = x1; x1 = x2; x2 = x4; x4 = x3; x3 = tmp;
			}
		} else if (parity & QUADRANT_ID[x3]) { // case 1(x3) 1(x4) 2(x1 x2)
			int tmp = x1; x1 = x3; x3 = tmp; tmp = x2; x2 = x4; x4 = tmp;
		}
	} else {
		if (!(parity & QUADRANT_ID[x2])) {
			if (parity & QUADRANT_ID[x3]) { // case 1(x1) 1(x3) 2(x2 x4)
				int tmp = x2; x2 = x3; x3 = tmp;
			} else { // case 1(x1) 1(x4) 2(x2 x3)
				int tmp = x2; x2 = x4; x4 = x3; x3 = tmp;
			}
		}
	}

	// best move alphabeta search
	bestscore = -SCORE_INF;
	if ((NEIGHBOUR[x1] & board->opponent) && board_next(board, x1, next)) {
		empty_remove(search->x_to_empties[x1]);
		bestscore = -search_solve_3(search, -(alpha + 1), next, parity ^ QUADRANT_ID[x1]);
		empty_restore(search->x_to_empties[x1]);
		if (bestscore > alpha) return bestscore;
	}

	if ((NEIGHBOUR[x2] & board->opponent) && board_next(board, x2, next)) {
		empty_remove(search->x_to_empties[x2]);
		score = -search_solve_3(search, -(alpha + 1), next, parity ^ QUADRANT_ID[x2]);
		empty_restore(search->x_to_empties[x2]);
		if (score > alpha) return score;
		else if (score > bestscore) bestscore = score;
	}

	if ((NEIGHBOUR[x3] & board->opponent) && board_next(board, x3, next)) {
		empty_remove(search->x_to_empties[x3]);
		score = -search_solve_3(search, -(alpha + 1), next, parity ^ QUADRANT_ID[x3]);
		empty_restore(search->x_to_empties[x3]);
		if (score > alpha) return score;
		else if (score > bestscore) bestscore = score;
	}

	if ((NEIGHBOUR[x4] & board->opponent) && board_next(board, x4, next)) {
		empty_remove(search->x_to_empties[x4]);
		score = -search_solve_3(search, -(alpha + 1), next, parity ^ QUADRANT_ID[x4]);
		empty_restore(search->x_to_empties[x4]);
		if (score > bestscore) bestscore = score;
	}

	// no move
	if (bestscore == -SCORE_INF) {
		if (can_move(board->opponent, board->player)) { // pass
			search_pass_endgame(search);
			bestscore = -search_solve_4(search, -(alpha + 1));
			search_pass_endgame(search);
		} else { // gameover
			bestscore = board_solve(board, 4);
		}
	}

 	assert(SCORE_MIN <= bestscore && bestscore <= SCORE_MAX);
	return bestscore;
}
#endif

/**
 * @brief  Evaluate a position using a shallow NWS.
 *
 * This function is used when there are few empty squares on the board. Here,
 * optimizations are in favour of speed instead of efficiency.
 * Move ordering is constricted to the hole parity and the type of squares.
 * No hashtable are used and anticipated cut-off is limited to stability cut-off.
 *
 * @param search Search.
 * @param alpha Alpha bound.
 * @return The final score, as a disc difference.
 */
static int search_shallow(Search *search, const int alpha)
{
	Board *board = search->board;
	SquareList *empty;
	Move move;
	int score, bestscore = -SCORE_INF;
	// const int beta = alpha + 1;

	assert(SCORE_MIN <= alpha && alpha <= SCORE_MAX);
	assert(0 <= search->n_empties && search->n_empties <= DEPTH_TO_SHALLOW_SEARCH);

	SEARCH_STATS(++statistics.n_NWS_shallow);
	SEARCH_UPDATE_INTERNAL_NODES(search->n_nodes);

	// stability cutoff
	if (search_SC_NWS(search, alpha, &score)) return score;

	if (search->parity > 0 && search->parity < 15) {

		foreach_odd_empty (empty, search->empties, search->parity) {
			if ((NEIGHBOUR[empty->x] & board->opponent)
			&& board_get_move(board, empty->x, &move)) {
				search_update_endgame(search, &move);
					if (search->n_empties == 4) score = -search_solve_4(search, -(alpha + 1));
					else score = -search_shallow(search, -(alpha + 1));
				search_restore_endgame(search, &move);
				if (score > alpha) return score;
				else if (score > bestscore) bestscore = score;
			}
		}

		foreach_even_empty (empty, search->empties, search->parity) {
			if ((NEIGHBOUR[empty->x] & board->opponent)
			&& board_get_move(board, empty->x, &move)) {
				search_update_endgame(search, &move);
					if (search->n_empties == 4) score = -search_solve_4(search, -(alpha + 1));
					else score = -search_shallow(search, -(alpha + 1));
				search_restore_endgame(search, &move);
				if (score > alpha) return score;
				else if (score > bestscore) bestscore = score;
			}
		}
	} else 	{
		foreach_empty (empty, search->empties) {
			if ((NEIGHBOUR[empty->x] & board->opponent)
			&& board_get_move(board, empty->x, &move)) {
				search_update_endgame(search, &move);
					if (search->n_empties == 4) score = -search_solve_4(search, -(alpha + 1));
					else score = -search_shallow(search, -(alpha + 1));
				search_restore_endgame(search, &move);
				if (score > alpha) return score;
				else if (score > bestscore) bestscore = score;
			}
		}
	}

	// no move
	if (bestscore == -SCORE_INF) {
		if (can_move(board->opponent, board->player)) { // pass
			search_pass_endgame(search);
				bestscore = -search_shallow(search, -(alpha + 1));
			search_pass_endgame(search);
		} else { // gameover
			bestscore = search_solve(search);
		}
	}

 	assert(SCORE_MIN <= bestscore && bestscore <= SCORE_MAX);
	return bestscore;
}

/**
 * @brief Evaluate an endgame position with a Null Window Search algorithm.
 *
 * This function is used when there are still many empty squares on the board. Move
 * ordering, hash table cutoff, enhanced transposition cutoff, etc. are used in
 * order to diminish the size of the tree to analyse, but at the expense of a
 * slower speed.
 *
 * @param search Search.
 * @param alpha Alpha bound.
 * @return The final score, as a disc difference.
 */
int NWS_endgame(Search *search, const int alpha)
{
	int score;
	HashTable *hash_table = search->hash_table;
	unsigned long long hash_code;
	// const int beta = alpha + 1;
	HashData hash_data[1];
	Board *board = search->board;
	MoveList movelist[1];
	Move *move, *bestmove;
	long long cost;

	if (search->stop) return alpha;

	assert(search->n_empties == bit_count(~(search->board->player|search->board->opponent)));
	assert(SCORE_MIN <= alpha && alpha <= SCORE_MAX);

	SEARCH_STATS(++statistics.n_NWS_endgame);

	if (search->n_empties <= DEPTH_TO_SHALLOW_SEARCH) return search_shallow(search, alpha);

	SEARCH_UPDATE_INTERNAL_NODES(search->n_nodes);

	// stability cutoff
	if (search_SC_NWS(search, alpha, &score)) return score;

	// transposition cutoff
	hash_code = board_get_hash_code(board);
	if (hash_get(hash_table, board, hash_code, hash_data) && search_TC_NWS(hash_data, search->n_empties, NO_SELECTIVITY, alpha, &score)) return score;

	search_get_movelist(search, movelist);

	cost = -search->n_nodes;

	// special cases
	if (movelist_is_empty(movelist)) {
		bestmove = movelist->move->next = movelist->move + 1;
		bestmove->next = 0;
		if (can_move(board->opponent, board->player)) { // pass
			search_pass_endgame(search);
				bestmove->score = -NWS_endgame(search, -(alpha + 1));
			search_pass_endgame(search);
			bestmove->x = PASS;
		} else  { // game over
			bestmove->score = search_solve(search);
			bestmove->x = NOMOVE;
		}
	} else {
		movelist_evaluate(movelist, search, hash_data, alpha, 0);

		bestmove = movelist->move; bestmove->score = -SCORE_INF;
		// loop over all moves
		foreach_best_move(move, movelist) {
			search_update_endgame(search, move);
				move->score = -NWS_endgame(search, -(alpha + 1));
			search_restore_endgame(search, move);
			if (move->score > bestmove->score) {
				bestmove = move;
				if (bestmove->score > alpha) break;
			}
		}
	}

	if (!search->stop) {
		cost += search->n_nodes;
		hash_store(hash_table, board, hash_code, search->n_empties, NO_SELECTIVITY, last_bit(cost), alpha, alpha + 1, bestmove->score, bestmove->x);
		if (SQUARE_STATS(1) + 0) {
			foreach_move(move, movelist)
				++statistics.n_played_square[search->n_empties][SQUARE_TYPE[move->x]];
			if (bestmove->score > alpha) ++statistics.n_good_square[search->n_empties][SQUARE_TYPE[bestmove->score]];
		}
	 	assert(SCORE_MIN <= bestmove->score && bestmove->score <= SCORE_MAX);
	 	assert((bestmove->score & 1) == 0);
		return bestmove->score;
	}

	return alpha;
}

