/**
 * @file board.c
 *
 * This module deals with the Board management.
 *
 * The Board is represented with a structure containing the following data:
 *  - a bitboard with the current player's square.
 *  - a bitboard with the current opponent's square.
 *
 * High level functions are provided to set/modify the board data or to compute
 * some board properties. Most of the functions are optimized to be as fast as
 * possible, while remaining readable.
 *
 * @date 1998 - 2018
 * @author Richard Delorme
 * @author Toshihiko Okuhara
 * @version 4.4
 */

#include "board.h"

#include "bit.h"
#include "hash.h"
#include "move.h"
#include "util.h"

#include <ctype.h>
#include <stdlib.h>
#include <assert.h>


#if MOVE_GENERATOR == MOVE_GENERATOR_CARRY
	#include "flip_carry_64.c"
#elif MOVE_GENERATOR == MOVE_GENERATOR_SSE
	#include "flip_sse.c"
#elif MOVE_GENERATOR == MOVE_GENERATOR_BITSCAN
	#include "flip_bitscan.c"
#elif MOVE_GENERATOR == MOVE_GENERATOR_ROXANE
	#include "flip_roxane.c"
#elif MOVE_GENERATOR == MOVE_GENERATOR_32
	#include "flip_carry_sse_32.c"
#elif MOVE_GENERATOR == MOVE_GENERATOR_SSE_BSWAP
	#include "flip_sse_bswap.c"
#elif MOVE_GENERATOR == MOVE_GENERATOR_AVX
	#include "flip_avx.c"
#else // MOVE_GENERATOR == MOVE_GENERATOR_KINDERGARTEN
	#include "flip_kindergarten.c"
#endif


/** edge stability global data */
unsigned char edge_stability[256 * 256];

/** conversion from an 8-bit line to the A1-A8 line */
const unsigned long long A1_A8[256] = {
	0x0000000000000000, 0x0000000000000001, 0x0000000000000100, 0x0000000000000101, 0x0000000000010000, 0x0000000000010001, 0x0000000000010100, 0x0000000000010101,
	0x0000000001000000, 0x0000000001000001, 0x0000000001000100, 0x0000000001000101, 0x0000000001010000, 0x0000000001010001, 0x0000000001010100, 0x0000000001010101,
	0x0000000100000000, 0x0000000100000001, 0x0000000100000100, 0x0000000100000101, 0x0000000100010000, 0x0000000100010001, 0x0000000100010100, 0x0000000100010101,
	0x0000000101000000, 0x0000000101000001, 0x0000000101000100, 0x0000000101000101, 0x0000000101010000, 0x0000000101010001, 0x0000000101010100, 0x0000000101010101,
	0x0000010000000000, 0x0000010000000001, 0x0000010000000100, 0x0000010000000101, 0x0000010000010000, 0x0000010000010001, 0x0000010000010100, 0x0000010000010101,
	0x0000010001000000, 0x0000010001000001, 0x0000010001000100, 0x0000010001000101, 0x0000010001010000, 0x0000010001010001, 0x0000010001010100, 0x0000010001010101,
	0x0000010100000000, 0x0000010100000001, 0x0000010100000100, 0x0000010100000101, 0x0000010100010000, 0x0000010100010001, 0x0000010100010100, 0x0000010100010101,
	0x0000010101000000, 0x0000010101000001, 0x0000010101000100, 0x0000010101000101, 0x0000010101010000, 0x0000010101010001, 0x0000010101010100, 0x0000010101010101,
	0x0001000000000000, 0x0001000000000001, 0x0001000000000100, 0x0001000000000101, 0x0001000000010000, 0x0001000000010001, 0x0001000000010100, 0x0001000000010101,
	0x0001000001000000, 0x0001000001000001, 0x0001000001000100, 0x0001000001000101, 0x0001000001010000, 0x0001000001010001, 0x0001000001010100, 0x0001000001010101,
	0x0001000100000000, 0x0001000100000001, 0x0001000100000100, 0x0001000100000101, 0x0001000100010000, 0x0001000100010001, 0x0001000100010100, 0x0001000100010101,
	0x0001000101000000, 0x0001000101000001, 0x0001000101000100, 0x0001000101000101, 0x0001000101010000, 0x0001000101010001, 0x0001000101010100, 0x0001000101010101,
	0x0001010000000000, 0x0001010000000001, 0x0001010000000100, 0x0001010000000101, 0x0001010000010000, 0x0001010000010001, 0x0001010000010100, 0x0001010000010101,
	0x0001010001000000, 0x0001010001000001, 0x0001010001000100, 0x0001010001000101, 0x0001010001010000, 0x0001010001010001, 0x0001010001010100, 0x0001010001010101,
	0x0001010100000000, 0x0001010100000001, 0x0001010100000100, 0x0001010100000101, 0x0001010100010000, 0x0001010100010001, 0x0001010100010100, 0x0001010100010101,
	0x0001010101000000, 0x0001010101000001, 0x0001010101000100, 0x0001010101000101, 0x0001010101010000, 0x0001010101010001, 0x0001010101010100, 0x0001010101010101,
	0x0100000000000000, 0x0100000000000001, 0x0100000000000100, 0x0100000000000101, 0x0100000000010000, 0x0100000000010001, 0x0100000000010100, 0x0100000000010101,
	0x0100000001000000, 0x0100000001000001, 0x0100000001000100, 0x0100000001000101, 0x0100000001010000, 0x0100000001010001, 0x0100000001010100, 0x0100000001010101,
	0x0100000100000000, 0x0100000100000001, 0x0100000100000100, 0x0100000100000101, 0x0100000100010000, 0x0100000100010001, 0x0100000100010100, 0x0100000100010101,
	0x0100000101000000, 0x0100000101000001, 0x0100000101000100, 0x0100000101000101, 0x0100000101010000, 0x0100000101010001, 0x0100000101010100, 0x0100000101010101,
	0x0100010000000000, 0x0100010000000001, 0x0100010000000100, 0x0100010000000101, 0x0100010000010000, 0x0100010000010001, 0x0100010000010100, 0x0100010000010101,
	0x0100010001000000, 0x0100010001000001, 0x0100010001000100, 0x0100010001000101, 0x0100010001010000, 0x0100010001010001, 0x0100010001010100, 0x0100010001010101,
	0x0100010100000000, 0x0100010100000001, 0x0100010100000100, 0x0100010100000101, 0x0100010100010000, 0x0100010100010001, 0x0100010100010100, 0x0100010100010101,
	0x0100010101000000, 0x0100010101000001, 0x0100010101000100, 0x0100010101000101, 0x0100010101010000, 0x0100010101010001, 0x0100010101010100, 0x0100010101010101,
	0x0101000000000000, 0x0101000000000001, 0x0101000000000100, 0x0101000000000101, 0x0101000000010000, 0x0101000000010001, 0x0101000000010100, 0x0101000000010101,
	0x0101000001000000, 0x0101000001000001, 0x0101000001000100, 0x0101000001000101, 0x0101000001010000, 0x0101000001010001, 0x0101000001010100, 0x0101000001010101,
	0x0101000100000000, 0x0101000100000001, 0x0101000100000100, 0x0101000100000101, 0x0101000100010000, 0x0101000100010001, 0x0101000100010100, 0x0101000100010101,
	0x0101000101000000, 0x0101000101000001, 0x0101000101000100, 0x0101000101000101, 0x0101000101010000, 0x0101000101010001, 0x0101000101010100, 0x0101000101010101,
	0x0101010000000000, 0x0101010000000001, 0x0101010000000100, 0x0101010000000101, 0x0101010000010000, 0x0101010000010001, 0x0101010000010100, 0x0101010000010101,
	0x0101010001000000, 0x0101010001000001, 0x0101010001000100, 0x0101010001000101, 0x0101010001010000, 0x0101010001010001, 0x0101010001010100, 0x0101010001010101,
	0x0101010100000000, 0x0101010100000001, 0x0101010100000100, 0x0101010100000101, 0x0101010100010000, 0x0101010100010001, 0x0101010100010100, 0x0101010100010101,
	0x0101010101000000, 0x0101010101000001, 0x0101010101000100, 0x0101010101000101, 0x0101010101010000, 0x0101010101010001, 0x0101010101010100, 0x0101010101010101,
};

#if defined(USE_GAS_MMX) || defined(USE_MSVC_X86)
#include "board_mmx.c"
#endif
#if defined(USE_GAS_MMX) || defined(USE_MSVC_X86) || defined(hasSSE2)
#include "board_sse.c"
#endif


/**
 * @brief Swap players.
 *
 * Swap players, i.e. change player's turn.
 *
 * @param board board
 */
void board_swap_players(Board *board)
{
	const unsigned long long tmp = board->player;
	board->player = board->opponent;
	board->opponent = tmp;
}

/**
 * @brief Set a board from a string description.
 *
 * Read a standardized string (See http://www.nada.kth.se/~gunnar/download2.html
 * for details) and translate it into our internal Board structure.
 *
 * @param board the board to set
 * @param string string describing the board
 * @return turn's color.
 */
int board_set(Board *board, const char *s)
{
	int i;
	unsigned long long b = 1;

	board->player = board->opponent = 0;
	for (i = A1; (i <= H8) && (*s != '\0'); ++s) {
		switch (tolower(*s)) {
		case 'b':
		case 'x':
		case '*':
			board->player |= b;
			break;
		case 'o':
		case 'w':
			board->opponent |= b;
			break;
		case '-':
		case '.':
			break;
		default:
			continue;
		}
		++i;
		b <<= 1;
	}
	board_check(board);

	for (; *s != '\0'; ++s) {
		switch (tolower(*s)) {
		case 'b':
		case 'x':
		case '*':
			return BLACK;
		case 'o':
		case 'w':
			board_swap_players(board);
			return WHITE;
		default:
			break;
		}
	}

	warn("board_set: bad string input\n");
	return EMPTY;
}

/**
 * @brief Set a board from a string description.
 *
 * Read a Forsyth-Edwards Notation string and translate it into our
 * internal Board structure.
 *
 * @param board the board to set
 * @param string string describing the board
 * @return turn's color.
 */
int board_from_FEN(Board *board, const char *string)
{
	int i;
	const char *s;

	board->player = board->opponent = 0;
	i = A8;
	for (s = parse_skip_spaces(string); *s && *s != ' '; ++s) {
		if (isdigit(*s)) {
			i += (*s - '0');
		} else if (*s == '/') {
			if (i & 7) return EMPTY;
			i -= 16;
		} else if (*s == 'p') {
			board->player |= x_to_bit(i);
			++i;
		} else if (*s == 'P') {
			board->opponent |= x_to_bit(i);
			++i;
		} else {
			return EMPTY;
		}
	}

	s = parse_skip_spaces(s);
	if (*s == 'b') {
		return BLACK;
	} else if (*s == 'w') {
		board_swap_players(board);
		return WHITE;
	}

	return EMPTY;
}

/**
 * @brief Set a board to the starting position.
 *
 * @param board the board to initialize
 */
void board_init(Board *board)
{
	board->player   = 0x0000000810000000; // BLACK
	board->opponent = 0x0000001008000000; // WHITE
}

/**
 * @brief Check board consistency
 *
 * @param board the board to initialize
 */
void board_check(const Board *board)
{
#ifndef NDEBUG
	if (board->player & board->opponent) {
		error("Two discs on the same square?\n");
		board_print(board, BLACK, stderr);
		bitboard_write(board->player, stderr);
		bitboard_write(board->opponent, stderr);
		abort();
	}

	// empty center ?
	if (~(board->player|board->opponent) & 0x0000001818000000) {
		error("Empty center?\n");
		board_print(board, BLACK, stderr);
	}
#else
	(void) board;
#endif // NDEBUG
}

/**
 * @brief Compare two board
 *
 * @param b1 first board
 * @param b2 second board
 * @return -1, 0, 1
 */
int board_compare(const Board *b1, const Board *b2)
{
	if (b1->player > b2->player) return 1;
	else if (b1->player < b2->player) return -1;
	else if (b1->opponent > b2->opponent) return 1;
	else if (b1->opponent < b2->opponent) return -1;
	else return 0;
}

/**
 * @brief Compare two board for equality
 *
 * @param b1 first board
 * @param b2 second board
 * @return true if both board are equal
 */
bool board_equal(const Board *b1, const Board *b2)
{
	return (b1->player == b2->player && b1->opponent == b2->opponent);
}

#ifndef hasSSE2	// SSE version in board_sse.c
/**
 * @brief symetric board
 *
 * @param board input board
 * @param s symetry
 * @param sym symetric output board
 */
void board_symetry(const Board *board, const int s, Board *sym)
{
	unsigned long long player, opponent;

	player = board->player;
	opponent = board->opponent;

	if (s & 1) {
		player = horizontal_mirror(player);
		opponent = horizontal_mirror(opponent);
	}
	if (s & 2) {
		player = vertical_mirror(player);
		opponent = vertical_mirror(opponent);
	}
	if (s & 4) {
		player = transpose(player);
		opponent = transpose(opponent);
	}

	sym->player = player;
	sym->opponent = opponent;

	board_check(sym);
}
#endif

/**
 * @brief unique board
 *
 * Compute a board unique from all its possible symertries.
 *
 * @param board input board
 * @param unique output board
 */
int board_unique(const Board *board, Board *unique)
{
	Board sym[8];
	int i, s = 0;

	assert(board != unique);

	*unique = *board;
	board_symetry(board,   1, &sym[1]);
	board_symetry(board,   2, &sym[2]);
	board_symetry(&sym[1], 2, &sym[3]);
	board_symetry(board,   4, &sym[4]);
	board_symetry(&sym[4], 2, &sym[5]);	// v-h reverted
	board_symetry(&sym[4], 1, &sym[6]);
	board_symetry(&sym[6], 2, &sym[7]);

	for (i = 1; i < 8; ++i) {
		// board_symetry(board, i, &sym);	// moved to before loop to minimize symetry ops
		if (board_compare(&sym[i], unique) < 0) {
			*unique = sym[i];
			s = i;
		}
	}

	board_check(unique);
	return s;
}

/** 
 * @brief Get a random board by playing random moves.
 * 
 * @param board The output board.
 * @param n_ply The number of random move to generate.
 * @param r The random generator.
 */
void board_rand(Board *board, int n_ply, Random *r)
{
	Move move[1];
	unsigned long long moves;
	int ply;

	board_init(board);
	for (ply = 0; ply < n_ply; ply++) {
		moves = get_moves(board->player, board->opponent);
		if (!moves) {
			board_pass(board);
			moves = get_moves(board->player, board->opponent);
			if (!moves) {
				break;
			}
		}
		board_get_move(board, get_rand_bit(moves, r), move);
		board_update(board, move);
	}
}


/**
 * @brief Compute a move.
 *
 * Compute how the board will be modified by a move without playing it.
 *
 * @param board board
 * @param x     square on which to move.
 * @param move  a Move structure remembering the modification.
 * @return      the flipped discs.
 */
unsigned long long board_get_move(const Board *board, const int x, Move *move)
{
	move->flipped = board_flip(board, x);
	move->x = x;
	return move->flipped;
}

/**
 * @brief Check if a move is legal.
 *
 * @param board board
 * @param move  a Move.
 * @return      true if the move is legal, false otherwise.
 */
bool board_check_move(const Board *board, Move *move)
{
	if (move->x == PASS) return !can_move(board->player, board->opponent);
	else if (x_to_bit(move->x) & (board->player | board->opponent)) return false;
	else if (move->flipped != board_flip(board, move->x)) return false;
	else return true;
}

#if !(defined(hasMMX) && (defined(USE_GAS_MMX) || defined(USE_MSVC_X86)))	// 32bit MMX/SSE version in board_mmx.c
/**
 * @brief Update a board.
 *
 * Update a board by flipping its discs and updating every other data,
 * according to the 'move' description.
 *
 * @param board the board to modify
 * @param move  A Move structure describing the modification.
 */
void board_update(Board *board, const Move *move)
{
	board->player ^= (move->flipped | x_to_bit(move->x));
	board->opponent ^= move->flipped;
	board_swap_players(board);
	board_check(board);
}

/**
 * @brief Restore a board.
 *
 * Restore a board by un-flipping its discs and restoring every other data,
 * according to the 'move' description, in order to cancel a board_update_move.
 *
 * @param board board to restore.
 * @param move  a Move structure describing the modification.
 */
void board_restore(Board *board, const Move *move)
{
	board_swap_players(board);
	board->player ^= (move->flipped | x_to_bit(move->x));
	board->opponent ^= move->flipped;
	board_check(board);
}
#endif // hasMMX

/**
 * @brief Passing move
 *
 * Modify a board by passing player's turn.
 *
 * @param board board to update.
 */
void board_pass(Board *board)
{
	board_swap_players(board);

	board_check(board);
}

#if !(defined(hasSSE2) && ((MOVE_GENERATOR == MOVE_GENERATOR_AVX) || (MOVE_GENERATOR == MOVE_GENERATOR_SSE)))	// SSE version in endgame_sse.c
/**
 * @brief Compute a board resulting of a move played on a previous board.
 *
 * @param board board to play the move on.
 * @param x move to play.
 * @param next resulting board.
 * @return flipped discs.
 */
unsigned long long board_next(const Board *board, const int x, Board *next)
{
	const unsigned long long flipped = board_flip(board, x);
	const unsigned long long player = board->opponent ^ flipped;

	next->opponent = board->player ^ (flipped | x_to_bit(x));
	next->player = player;

	return flipped;
}

/**
 * @brief Compute a board resulting of an opponent move played on a previous board.
 *
 * Compute the board after passing and playing a move.
 *
 * @param board board to play the move on.
 * @param x opponent move to play.
 * @param next resulting board.
 * @return flipped discs.
 */
unsigned long long board_pass_next(const Board *board, const int x, Board *next)
{
	const unsigned long long flipped = Flip(x, board->opponent, board->player);

	next->opponent = board->opponent ^ (flipped | x_to_bit(x));
	next->player = board->player ^ flipped;

	return flipped;
}
#endif

#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__AVX2__)	// sse version in board_sse.c
/**
 * @brief Get a part of the moves.
 *
 * Partially compute a bitboard where each coordinate with a legal move is set to one.
 *
 * Two variants of the algorithm are provided, one based on Kogge-Stone parallel
 * prefix.
 *
 * @param P bitboard with player's discs.
 * @param mask bitboard with flippable opponent's discs.
 * @param dir flipping direction.
 * @return some legal moves in a 64-bit unsigned integer.
 */
static inline unsigned long long get_some_moves(const unsigned long long P, const unsigned long long mask, const int dir)
// x86 build will use helper for long long shift unless inlined
{

#if KOGGE_STONE & 1
	// kogge-stone algorithm
 	// 6 << + 6 >> + 12 & + 7 |
	// + better instruction independency
	unsigned long long flip_l, flip_r;
	unsigned long long mask_l, mask_r;
	int d;

	flip_l = flip_r = P;
	mask_l = mask_r = mask;
	d = dir;

	flip_l |= mask_l & (flip_l << d);   flip_r |= mask_r & (flip_r >> d);
	mask_l &= (mask_l << d);            mask_r &= (mask_r >> d);
	d <<= 1;
	flip_l |= mask_l & (flip_l << d);   flip_r |= mask_r & (flip_r >> d);
	mask_l &= (mask_l << d);            mask_r &= (mask_r >> d);
	d <<= 1;
	flip_l |= mask_l & (flip_l << d);   flip_r |= mask_r & (flip_r >> d);

	return ((flip_l & mask) << dir) | ((flip_r & mask) >> dir);

#elif PARALLEL_PREFIX & 1
	// 1-stage Parallel Prefix (intermediate between kogge stone & sequential) 
	// 6 << + 6 >> + 7 | + 10 &
	unsigned long long flip_l, flip_r;
	unsigned long long mask_l, mask_r;
	const int dir2 = dir + dir;

	flip_l  = mask & (P << dir);          flip_r  = mask & (P >> dir);
	flip_l |= mask & (flip_l << dir);     flip_r |= mask & (flip_r >> dir);
	mask_l  = mask & (mask << dir);       mask_r  = mask_l >> dir;
	flip_l |= mask_l & (flip_l << dir2);  flip_r |= mask_r & (flip_r >> dir2);
	flip_l |= mask_l & (flip_l << dir2);  flip_r |= mask_r & (flip_r >> dir2);

	return (flip_l << dir) | (flip_r >> dir);

#else
 	// sequential algorithm
 	// 7 << + 7 >> + 6 & + 12 |
	unsigned long long flip;

	flip = (((P << dir) | (P >> dir)) & mask);
	flip |= (((flip << dir) | (flip >> dir)) & mask);
	flip |= (((flip << dir) | (flip >> dir)) & mask);
	flip |= (((flip << dir) | (flip >> dir)) & mask);
	flip |= (((flip << dir) | (flip >> dir)) & mask);
	flip |= (((flip << dir) | (flip >> dir)) & mask);
	return (flip << dir) | (flip >> dir);

#endif
}

/**
 * @brief Get legal moves.
 *
 * Compute a bitboard where each coordinate with a legal move is set to one.
 *
 * @param P bitboard with player's discs.
 * @param O bitboard with opponent's discs.
 * @return all legal moves in a 64-bit unsigned integer.
 */
unsigned long long get_moves(const unsigned long long P, const unsigned long long O)
{
	unsigned long long moves, OM;

	#if defined(USE_GAS_MMX) || defined(USE_MSVC_X86)
	if (hasSSE2)
		return get_moves_sse((unsigned int) P, (unsigned int) (P >> 32), (unsigned int) O, (unsigned int) (O >> 32));
	else if (hasMMX)
		return get_moves_mmx((unsigned int) P, (unsigned int) (P >> 32), (unsigned int) O, (unsigned int) (O >> 32));
	#endif

	OM = O & 0x7e7e7e7e7e7e7e7e;
	moves = ( get_some_moves(P, OM, 1) // horizontal
		| get_some_moves(P, O, 8)   // vertical
		| get_some_moves(P, OM, 7)   // diagonals
		| get_some_moves(P, OM, 9));

	return moves & ~(P|O);	// mask with empties
}
#endif

/**
 * @brief Get legal moves on a 6x6 board.
 *
 * Compute a bitboard where each coordinate with a legal move is set to one.
 *
 * @param P bitboard with player's discs.
 * @param O bitboard with opponent's discs.
 * @return all legal moves in a 64-bit unsigned integer.
 */
unsigned long long get_moves_6x6(const unsigned long long P, const unsigned long long O)
{
	return get_moves(P & 0x007E7E7E7E7E7E00, O & 0x007E7E7E7E7E7E00) & 0x007E7E7E7E7E7E00;
}

/**
 * @brief Check if a player can move.
 *
 * @param P bitboard with player's discs.
 * @param O bitboard with opponent's discs.
 * @return true or false.
 */
bool can_move(const unsigned long long P, const unsigned long long O)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(hasMMX)
	return get_moves(P, O) != 0;

#else
	const unsigned long long E = ~(P|O); // empties
	const unsigned long long OM = O & 0x7E7E7E7E7E7E7E7E;

	return (get_some_moves(P, OM, 7) & E)  // diagonals
		|| (get_some_moves(P, OM, 9) & E)
		|| (get_some_moves(P, OM, 1) & E)  // horizontal
		|| (get_some_moves(P, O, 8) & E); // vertical
#endif
}

/**
 * @brief Check if a player can move.
 *
 * @param P bitboard with player's discs.
 * @param O bitboard with opponent's discs.
 * @return true or false.
 */
bool can_move_6x6(const unsigned long long P, const unsigned long long O)
{
	return get_moves_6x6(P, O) != 0;
}

/**
 * @brief Count legal moves.
 *
 * Compute mobility, ie the number of legal moves.
 *
 * @param P bitboard with player's discs.
 * @param O bitboard with opponent's discs.
 * @return a count of all legal moves.
 */
int get_mobility(const unsigned long long P, const unsigned long long O)
{
	return bit_count(get_moves(P, O));
}

int get_weighted_mobility(const unsigned long long P, const unsigned long long O)
{
	return bit_weighted_count(get_moves(P, O));
}

/**
 * @brief Get some potential moves.
 *
 * @param O bitboard with opponent's discs.
 * @param dir flipping direction.
 * @return some potential moves in a 64-bit unsigned integer.
 */
static inline unsigned long long get_some_potential_moves(const unsigned long long O, const int dir)
{
	return (O << dir | O >> dir);
}

/**
 * @brief Get potential moves.
 *
 * Get the list of empty squares in contact of a player square.
 *
 * @param P bitboard with player's discs.
 * @param O bitboard with opponent's discs.
 * @return all potential moves in a 64-bit unsigned integer.
 */
static unsigned long long get_potential_moves(const unsigned long long P, const unsigned long long O)
{
	return (get_some_potential_moves(O & 0x7E7E7E7E7E7E7E7E, 1) // horizontal
		| get_some_potential_moves(O & 0x00FFFFFFFFFFFF00, 8)   // vertical
		| get_some_potential_moves(O & 0x007E7E7E7E7E7E00, 7)   // diagonals
		| get_some_potential_moves(O & 0x007E7E7E7E7E7E00, 9))
		& ~(P|O); // mask with empties
}

/**
 * @brief Get potential mobility.
 *
 * Count the list of empty squares in contact of a player square.
 *
 * @param P bitboard with player's discs.
 * @param O bitboard with opponent's discs.
 * @return a count of potential moves.
 */
int get_potential_mobility(const unsigned long long P, const unsigned long long O)
{
#if defined(USE_GAS_MMX) || defined(USE_MSVC_X86)
	if (hasMMX)
		return get_potential_mobility_mmx(P, O);
#endif
	return bit_weighted_count(get_potential_moves(P, O));
}

/**
 * @brief search stable edge patterns.
 *
 * Compute a 8-bit bitboard where each stable square is set to one
 *
 * @param old_P previous player edge discs.
 * @param old_O previous opponent edge discs.
 * @param stable 8-bit bitboard with stable edge squares.
 */
static int find_edge_stable(const int old_P, const int old_O, int stable)
{
	int P, O, O2, X, F;
	const int E = ~(old_P | old_O); // empties

	stable &= old_P; // mask stable squares with remaining player squares.
	if (!stable || E == 0) return stable;

	for (X = 0x01; X <= 0x80; X <<= 1) {
		if (E & X) { // is x an empty square ?
			O = old_O;
			P = old_P | X; // player plays on it
			if (X > 0x02) { // flip left discs (using parallel prefix)
				F  = O & (X >> 1);
				F |= O & (F >> 1);
				O2 = O & (O >> 1);
				F |= O2 & (F >> 2);
				F |= O2 & (F >> 2);
				F &= -(P & (F >> 1));
				O ^= F;
				P ^= F;
			}
			// if (X < 0x40) { // flip right discs (using carry propagation)
				F = (O + X + X) & P;
				F -= (X + X) & -(int)(F != 0);
				O ^= F;
				P ^= F;
			// }
			stable = find_edge_stable(P, O, stable); // next move
			if (!stable) return stable;

			P = old_P;
			O = old_O | X; // opponent plays on it
			if (X > 0x02) { // flip left discs (using parallel prefix)
				F  = P & (X >> 1);
				F |= P & (F >> 1);
				O2 = P & (P >> 1);
				F |= O2 & (F >> 2);
				F |= O2 & (F >> 2);
				F &= -(O & (F >> 1));
				O ^= F;
				P ^= F;
			}
			// if (X < 0x40) { // flip right discs (using carry propagation)
	 			F = (P + X + X) & O;
				F -= (X + X) & -(int)(F != 0);
				O ^= F;
				P ^= F;
			// }
			stable = find_edge_stable(P, O, stable); // next move
			if (!stable) return stable;
		}
	}

	return stable;
}

/**
 * @brief Initialize the edge stability tables.
 */
void edge_stability_init(void)
{
	int P, O, PO, rPO;
	// long long t = cpu_clock();

	for (PO = 0; PO < 256 * 256; ++PO) {
		P = PO >> 8;
		O = PO & 0xFF;
		if (P & O) { // illegal positions
			edge_stability[PO] = 0;
		} else {
			rPO = horizontal_mirror_32(PO);
			if (PO > rPO)
				edge_stability[PO] = horizontal_mirror_32(edge_stability[rPO]);
			else
				edge_stability[PO] = find_edge_stable(P, O, P);
		}
	}
	// printf("edge_stability_init: %d\n", (int)(cpu_clock() - t));

#if (defined(USE_GAS_MMX) || defined(USE_MSVC_X86)) && !defined(hasSSE2)
	init_mmx();
#endif
}

#ifdef HAS_CPU_64
#define	packA1A8(X)	((((X) & 0x0101010101010101) * 0x0102040810204080) >> 56)
#define	packH1H8(X)	((((X) & 0x8080808080808080) * 0x0002040810204081) >> 56)
#else
#define	packA1A8(X)	(((((unsigned int)(X) & 0x01010101) + (((unsigned int)((X) >> 32) & 0x01010101) << 4)) * 0x01020408) >> 24)
#define	packH1H8(X)	(((((unsigned int)((X) >> 32) & 0x80808080) + (((unsigned int)(X) & 0x80808080) >> 4)) * 0x00204081) >> 24)
#endif

#if !defined(__x86_64__) && !defined(_M_X64)
/**
 * @brief Get full lines.
 *
 * @param line all discs on a line.
 * @param dir tested direction
 * @return a bitboard with f lines along the tested direction.
 */
static inline unsigned long long get_full_lines(const unsigned long long line, const int dir)
{
#if KOGGE_STONE & 2

	// kogge-stone algorithm
 	// 5 << + 5 >> + 7 & + 10 |
	// + better instruction independency
	unsigned long long full_l, full_r, edge_l, edge_r;
	const  unsigned long long edge = 0xff818181818181ff;
	const int dir2 = dir << 1;
	const int dir4 = dir << 2;

	full_l = line & (edge | (line >> dir)); full_r  = line & (edge | (line << dir));
	edge_l = edge | (edge >> dir);        edge_r  = edge | (edge << dir);
	full_l &= edge_l | (full_l >> dir2);  full_r &= edge_r | (full_r << dir2);
	edge_l |= edge_l >> dir2;             edge_r |= edge_r << dir2;
	full_l &= edge_l | (full_l >> dir4);  full_r &= edge_r | (full_r << dir4);

	return full_r & full_l;

#elif PARALLEL_PREFIX & 2

	// 1-stage Parallel Prefix (intermediate between kogge stone & sequential) 
	// 5 << + 5 >> + 7 & + 10 |
	unsigned long long full_l, full_r;
	unsigned long long edge_l, edge_r;
	const  unsigned long long edge = 0xff818181818181ff;
	const int dir2 = dir + dir;

	full_l  = edge | (line << dir);       full_r  = edge | (line >> dir);
	full_l &= edge | (full_l << dir);     full_r &= edge | (full_r >> dir);
	edge_l  = edge | (edge << dir);       edge_r  = edge | (edge >> dir);
	full_l &= edge_l | (full_l << dir2);  full_r &= edge_r | (full_r >> dir2);
	full_l &= edge_l | (full_l << dir2);  full_r &= edge_r | (full_r >> dir2);

	return full_l & full_r;

#else

	// sequential algorithm
 	// 6 << + 6 >> + 12 & + 5 |
	unsigned long long full;
	const unsigned long long edge = line & 0xff818181818181ff;

	full = (line & (((line >> dir) & (line << dir)) | edge));
	full &= (((full >> dir) & (full << dir)) | edge);
	full &= (((full >> dir) & (full << dir)) | edge);
	full &= (((full >> dir) & (full << dir)) | edge);
	full &= (((full >> dir) & (full << dir)) | edge);

	return ((full >> dir) & (full << dir));

#endif
}

#ifdef HAS_CPU_64
static unsigned long long get_full_lines_h(unsigned long long full)
{
	full &= full >> 1;
	full &= full >> 2;
	full &= full >> 4;
	return (full & 0x0101010101010101) * 0xff;
}
#else
static unsigned int get_full_lines_h_32(unsigned int full)
{
	full &= full >> 1;
	full &= full >> 2;
	full &= full >> 4;
	return (full & 0x01010101) * 0xff;
}

static unsigned long long get_full_lines_h(unsigned long long full)
{
	return ((unsigned long long) get_full_lines_h_32(full >> 32) << 32) | get_full_lines_h_32(full);
}
#endif

static unsigned long long get_full_lines_v(unsigned long long full)
{
#ifdef _MSC_VER
	full &= _rotr64(full, 8);
	full &= _rotr64(full, 16);
	full &= _rotr64(full, 32);
#else
	full &= (full >> 8) | (full << 56);	// ror 8
	full &= (full >> 16) | (full << 48);	// ror 16
	full &= (full >> 32) | (full << 32);	// ror 32
#endif
	return full;
}

/**
 * @brief Get stable edge.
 *
 * @param P bitboard with player's discs.
 * @param O bitboard with opponent's discs.
 * @return a bitboard with (some of) player's stable discs.
 *
 */
static unsigned long long get_stable_edge(const unsigned long long P, const unsigned long long O)
{	// compute the exact stable edges (from precomputed tables)
	return edge_stability[((unsigned int) P & 0xff) * 256 + ((unsigned int) O & 0xff)]
	    |  (unsigned long long) edge_stability[(unsigned int) (P >> 56) * 256 + (unsigned int) (O >> 56)] << 56
	    |  A1_A8[edge_stability[packA1A8(P) * 256 + packA1A8(O)]]
	    |  A1_A8[edge_stability[packH1H8(P) * 256 + packH1H8(O)]] << 7;
}

/**
 * @brief Estimate the stability.
 *
 * Count the number (in fact a lower estimate) of stable discs.
 *
 * @param P bitboard with player's discs.
 * @param O bitboard with opponent's discs.
 * @return the number of stable discs.
 */
int get_stability(const unsigned long long P, const unsigned long long O)
{
	unsigned long long P_central, disc, full_h, full_v, full_d7, full_d9;
	unsigned long long stable_h, stable_v, stable_d7, stable_d9, stable, old_stable;

#if (defined(USE_GAS_MMX) && !(defined(__clang__) && (__clang__major__ < 3))) || defined(USE_MSVC_X86)
	if (hasMMX)
		return get_stability_mmx((unsigned int) P, (unsigned int) (P >> 32), (unsigned int) O, (unsigned int) (O >> 32));
#endif

	disc = (P | O);
	P_central = (P & 0x007e7e7e7e7e7e00);

	full_h = get_full_lines_h(disc);
	full_v = get_full_lines_v(disc);
	full_d7 = get_full_lines(disc, 7);
	full_d9 = get_full_lines(disc, 9);

	// compute the exact stable edges (from precomputed tables)
	stable = get_stable_edge(P, O);

	// add full lines
	stable |= (full_h & full_v & full_d7 & full_d9 & P_central);

	if (stable == 0)
		return 0;

	// now compute the other stable discs (ie discs touching another stable disc in each flipping direction).
	do {
		old_stable = stable;
		stable_h = ((stable >> 1) | (stable << 1) | full_h);
		stable_v = ((stable >> 8) | (stable << 8) | full_v);
		stable_d7 = ((stable >> 7) | (stable << 7) | full_d7);
		stable_d9 = ((stable >> 9) | (stable << 9) | full_d9);
		stable |= (stable_h & stable_v & stable_d7 & stable_d9 & P_central);
	} while (stable != old_stable);

	return bit_count(stable);
}
#endif // __x86_64__

/**
 * @brief Estimate the stability of edges.
 *
 * Count the number (in fact a lower estimate) of stable discs on the edges.
 *
 * @param P bitboard with player's discs.
 * @param O bitboard with opponent's discs.
 * @return the number of stable discs on the edges.
 */
int get_edge_stability(const unsigned long long P, const unsigned long long O)
{
	return bit_count(get_stable_edge(P, O));
}

/**
 * @brief Estimate corner stability.
 *
 * Count the number of stable discs around the corner. Limiting the count
 * to the corner keep the function fast but still get this information,
 * particularly important at Othello. Corner stability will be used for
 * move sorting.
 *
 * @param P bitboard with player's discs.
 * @return the number of stable discs around the corner.
 */
int get_corner_stability(const unsigned long long P)
{
#if 0

	const unsigned long long stable = ((((0x0100000000000001 & P) << 1) | ((0x8000000000000080 & P) >> 1) | ((0x0000000000000081 & P) << 8) | ((0x8100000000000000 & P) >> 8) | 0x8100000000000081) & P);
	return bit_count(stable);

#else	// kindergarten

	static const char n_stable_h8g8b8a8h7a7[64] = {
		0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 2, 3, 2, 3,
		0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 2, 3, 2, 3,
		1, 1, 2, 2, 2, 3, 3, 4, 1, 1, 2, 2, 3, 4, 4, 5,
		2, 2, 3, 3, 3, 4, 4, 5, 2, 2, 3, 3, 4, 5, 5, 6
	};
	static const char n_stable_h2a2h1g1b1a1[64] = {
		0, 1, 0, 2, 0, 1, 0, 2, 1, 2, 1, 3, 2, 3, 2, 4,
		0, 2, 0, 3, 0, 2, 0, 3, 1, 3, 1, 4, 2, 4, 2, 5,
		0, 1, 0, 2, 0, 1, 0, 2, 2, 3, 2, 4, 3, 4, 3, 5,
		0, 2, 0, 3, 0, 2, 0, 3, 2, 4, 2, 5, 3, 5, 3, 6
	};

#if 0 // defined(__BMI2__) && defined(__x86_64__) // pext is slow on AMD
	int cnt = n_stable_h8g8b8a8h7a7[_pext_u64(P, 0xc381000000000000)]
		+ n_stable_h2a2h1g1b1a1[_pext_u32((unsigned int) P, 0x000081c3)];
#else
	int cnt = n_stable_h8g8b8a8h7a7[(((unsigned int) (P >> 32) & 0xc3810000) * 0x00000411) >> 26]
		+ n_stable_h2a2h1g1b1a1[(((unsigned int) P & 0x000081c3) * 0x04410000) >> 26];
#endif
	// assert(cnt == bit_count((((0x0100000000000001 & P) << 1) | ((0x8000000000000080 & P) >> 1) | ((0x0000000000000081 & P) << 8) | ((0x8100000000000000 & P) >> 8) | 0x8100000000000081) & P));
	return cnt;

#endif
}

/**
 * @brief Compute a hash code.
 *
 * @param board the board.
 * @return the hash code of the bitboard
 */
unsigned long long board_get_hash_code(const Board *board)
{
	const unsigned char *p = (const unsigned char*)board;
	unsigned long long h1, h2;

#if defined(USE_GAS_MMX) && defined(__3dNOW__)	// Faster on AMD but not suitable for CPU with slow emms
	if (hasMMX)
		return board_get_hash_code_mmx(p);
#elif defined(USE_GAS_MMX) || defined(USE_MSVC_X86) // || defined(__x86_64__)
	if (hasSSE2)
		return board_get_hash_code_sse(p);
#endif

	h1  = hash_rank[0][p[0]];	h2  = hash_rank[1][p[1]];
	h1 ^= hash_rank[2][p[2]];	h2 ^= hash_rank[3][p[3]];
	h1 ^= hash_rank[4][p[4]];	h2 ^= hash_rank[5][p[5]];
	h1 ^= hash_rank[6][p[6]];	h2 ^= hash_rank[7][p[7]];
	h1 ^= hash_rank[8][p[8]];	h2 ^= hash_rank[9][p[9]];
	h1 ^= hash_rank[10][p[10]];	h2 ^= hash_rank[11][p[11]];
	h1 ^= hash_rank[12][p[12]];	h2 ^= hash_rank[13][p[13]];
	h1 ^= hash_rank[14][p[14]];	h2 ^= hash_rank[15][p[15]];

	// assert((h1 ^ h2) == board_get_hash_code_sse(p));

	return h1 ^ h2;
}

/**
 * @brief Get square color.
 *
 * returned value: 0 = player, 1 = opponent, 2 = empty;
 *
 * @param board board.
 * @param x square coordinate.
 * @return square color.
 */
int board_get_square_color(const Board *board, const int x)
{
	unsigned long long b = x_to_bit(x);
	return (int) ((board->player & b) == 0) * 2 - (int) ((board->opponent & b) != 0);
}

/**
 * @brief Check if a square is occupied.
 *
 * @param board board.
 * @param x square coordinate.
 * @return true if a square is occupied.
 */
bool board_is_occupied(const Board *board, const int x)
{
	return ((board->player | board->opponent) & x_to_bit(x)) != 0;	// omitting != 0 causes bogus code on MSVC19 /GL
}

/**
 * @brief Check if current player should pass.
 *
 * @param board board.
 * @return true if player is passing, false otherwise.
 */
bool board_is_pass(const Board *board)
{
	return !can_move(board->player, board->opponent) &&
		can_move(board->opponent, board->player);
}

/**
 * @brief Check if the game is over.
 *
 * @param board board.
 * @return true if game is over, false otherwise.
 */
bool board_is_game_over(const Board *board)
{
	return !can_move(board->player, board->opponent) &&
		!can_move(board->opponent, board->player);
}


/**
 * @brief Check if the game is over.
 *
 * @param board board.
 * @return true if game is over, false otherwise.
 */
int board_count_empties(const Board *board)
{
	return bit_count(~(board->player | board->opponent));
}

/**
 * @brief Print out the board.
 *
 * Print an ASCII representation of the board to an output stream.
 *
 * @param board board to print.
 * @param player player's color.
 * @param f output stream.
 */
void board_print(const Board *board, const int player, FILE *f)
{
	int i, j, square;
	unsigned long long bk, wh;
	const char *color = "?*O-." + 1;
	unsigned long long moves = get_moves(board->player, board->opponent);

	if (player == BLACK) {
		bk = board->player;
		wh = board->opponent;
	} else {
		bk = board->opponent;
		wh = board->player;
	}

	fputs("  A B C D E F G H\n", f);
	for (i = 0; i < 8; ++i) {
		fputc(i + '1', f);
		fputc(' ', f);
		for (j = 0; j < 8; ++j) {
			square = 2 - (wh & 1) - 2 * (bk & 1);
			if ((square == EMPTY) && (moves & 1))
				square = EMPTY + 1;
			fputc(color[square], f);
			fputc(' ', f);
			bk >>= 1;
			wh >>= 1;
			moves >>= 1;
		}
		fputc(i + '1', f);
		if (i == 1)
			fprintf(f, " %c to move", color[player]);
		else if (i == 3)
			fprintf(f, " %c: discs = %2d    moves = %2d",
				color[player], bit_count(board->player), get_mobility(board->player, board->opponent));
		else if (i == 4)
			fprintf(f, " %c: discs = %2d    moves = %2d",
				color[!player], bit_count(board->opponent), get_mobility(board->opponent, board->player));
		else if (i == 5)
			fprintf(f, "  empties = %2d      ply = %2d",
				64 - bit_count(board->opponent|board->player), bit_count(board->opponent|board->player) - 3);
		fputc('\n', f);
	}
	fputs("  A B C D E F G H\n", f);
}

/**
 * @brief convert the to a compact string.
 *
 * @param board board to convert.
 * @param player player's color.
 * @param s output string.
 */
char* board_to_string(const Board *board, const int player, char *s)
{
	int square, x;
	unsigned long long bk, wh;
	static const char color[4] = "XO-?";

	if (player == BLACK) {
		bk = board->player;
		wh = board->opponent;
	} else {
		bk = board->opponent;
		wh = board->player;
	}

	for (x = 0; x < 64; ++x) {
		square = 2 - (wh & 1) - 2 * (bk & 1);
		s[x] = color[square];
		bk >>= 1;
		wh >>= 1;
	}
	s[64] = ' ';
	s[65] = color[player];
	s[66] = '\0';
	return s;
}

/**
 * @brief print using FEN description.
 *
 * Write the board according to the Forsyth-Edwards Notation.
 *
 * @param board the board to write
 * @param player turn's color.
 * @param f output stream.
 */
void board_print_FEN(const Board *board, const int player, FILE *f)
{
	char s[256];
	fputs(board_to_FEN(board, player, s), f);
}

/**
 * @brief print to FEN description.
 *
 * Write the board into a Forsyth-Edwards Notation string.
 *
 * @param board the board to write
 * @param player turn's color.
 * @param string output string.
 */
char* board_to_FEN(const Board *board, const int player, char *string)
{
	int square, x, r, c;
	unsigned long long bk, wh;
	static const char piece[4] = "pP-?";
	static const char color[2] = "bw";
	int n_empties = 0;
	char *s = string;
	static char local_string[128];

	if (s == NULL) s = string = local_string;

	if (player == BLACK) {
		bk = board->player;
		wh = board->opponent;
	} else {
		bk = board->opponent;
		wh = board->player;
	}

	for (r = 7; r >= 0; --r) {
		for (c = 0; c < 8; ++c) {
			x = 8 * r + c;
			square = 2 - ((wh >> x) & 1) - 2 * ((bk >> x) & 1);

			if (square == EMPTY) {
				++n_empties;
			} else {
				if (n_empties) {
					*s++ = n_empties + '0';
					n_empties = 0;
				}
				*s++ = piece[square];
			}
		}
		if (n_empties) {
			*s++ = n_empties + '0';
			n_empties = 0;
		}
		if (r > 0)
			*s++ = '/';
	}
	*s++ = ' ';
	*s++ = color[player];
	*s++ = ' '; *s++ = '-'; *s++ = ' '; *s++ = '-'; 
	*s++ = ' '; *s++ = '0'; *s++ = ' '; *s++ = '1'; *s = '\0';

	return string;
}

