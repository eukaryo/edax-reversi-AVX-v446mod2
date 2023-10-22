/**
 * @file bit.c
 *
 * Bitwise operations.
 * Several algorithms manipulating bits are presented here. Quite often,
 * a macro needs to be defined to chose between different flavors of the
 * algorithm.
 *
 * @date 1998 - 2018
 * @author Richard Delorme
 * @version 4.4
 */

#include "bit.h"
#include "util.h"

/** coordinate to bit table converter */
const unsigned long long X_TO_BIT[] = {
	0x0000000000000001ULL, 0x0000000000000002ULL, 0x0000000000000004ULL, 0x0000000000000008ULL,
	0x0000000000000010ULL, 0x0000000000000020ULL, 0x0000000000000040ULL, 0x0000000000000080ULL,
	0x0000000000000100ULL, 0x0000000000000200ULL, 0x0000000000000400ULL, 0x0000000000000800ULL,
	0x0000000000001000ULL, 0x0000000000002000ULL, 0x0000000000004000ULL, 0x0000000000008000ULL,
	0x0000000000010000ULL, 0x0000000000020000ULL, 0x0000000000040000ULL, 0x0000000000080000ULL,
	0x0000000000100000ULL, 0x0000000000200000ULL, 0x0000000000400000ULL, 0x0000000000800000ULL,
	0x0000000001000000ULL, 0x0000000002000000ULL, 0x0000000004000000ULL, 0x0000000008000000ULL,
	0x0000000010000000ULL, 0x0000000020000000ULL, 0x0000000040000000ULL, 0x0000000080000000ULL,
	0x0000000100000000ULL, 0x0000000200000000ULL, 0x0000000400000000ULL, 0x0000000800000000ULL,
	0x0000001000000000ULL, 0x0000002000000000ULL, 0x0000004000000000ULL, 0x0000008000000000ULL,
	0x0000010000000000ULL, 0x0000020000000000ULL, 0x0000040000000000ULL, 0x0000080000000000ULL,
	0x0000100000000000ULL, 0x0000200000000000ULL, 0x0000400000000000ULL, 0x0000800000000000ULL,
	0x0001000000000000ULL, 0x0002000000000000ULL, 0x0004000000000000ULL, 0x0008000000000000ULL,
	0x0010000000000000ULL, 0x0020000000000000ULL, 0x0040000000000000ULL, 0x0080000000000000ULL,
	0x0100000000000000ULL, 0x0200000000000000ULL, 0x0400000000000000ULL, 0x0800000000000000ULL,
	0x1000000000000000ULL, 0x2000000000000000ULL, 0x4000000000000000ULL, 0x8000000000000000ULL,
	0, 0 // <- hack for passing move & nomove
};

/** Conversion array: neighbour bits */
const unsigned long long NEIGHBOUR[] = {
	0x0000000000000302ULL, 0x0000000000000705ULL, 0x0000000000000e0aULL, 0x0000000000001c14ULL,
	0x0000000000003828ULL, 0x0000000000007050ULL, 0x000000000000e0a0ULL, 0x000000000000c040ULL,
	0x0000000000030203ULL, 0x0000000000070507ULL, 0x00000000000e0a0eULL, 0x00000000001c141cULL,
	0x0000000000382838ULL, 0x0000000000705070ULL, 0x0000000000e0a0e0ULL, 0x0000000000c040c0ULL,
	0x0000000003020300ULL, 0x0000000007050700ULL, 0x000000000e0a0e00ULL, 0x000000001c141c00ULL,
	0x0000000038283800ULL, 0x0000000070507000ULL, 0x00000000e0a0e000ULL, 0x00000000c040c000ULL,
	0x0000000302030000ULL, 0x0000000705070000ULL, 0x0000000e0a0e0000ULL, 0x0000001c141c0000ULL,
	0x0000003828380000ULL, 0x0000007050700000ULL, 0x000000e0a0e00000ULL, 0x000000c040c00000ULL,
	0x0000030203000000ULL, 0x0000070507000000ULL, 0x00000e0a0e000000ULL, 0x00001c141c000000ULL,
	0x0000382838000000ULL, 0x0000705070000000ULL, 0x0000e0a0e0000000ULL, 0x0000c040c0000000ULL,
	0x0003020300000000ULL, 0x0007050700000000ULL, 0x000e0a0e00000000ULL, 0x001c141c00000000ULL,
	0x0038283800000000ULL, 0x0070507000000000ULL, 0x00e0a0e000000000ULL, 0x00c040c000000000ULL,
	0x0302030000000000ULL, 0x0705070000000000ULL, 0x0e0a0e0000000000ULL, 0x1c141c0000000000ULL,
	0x3828380000000000ULL, 0x7050700000000000ULL, 0xe0a0e00000000000ULL, 0xc040c00000000000ULL,
	0x0203000000000000ULL, 0x0507000000000000ULL, 0x0a0e000000000000ULL, 0x141c000000000000ULL,
	0x2838000000000000ULL, 0x5070000000000000ULL, 0xa0e0000000000000ULL, 0x40c0000000000000ULL,
	0, 0 // <- hack for passing move & nomove
};

/**
 * @brief Count the number of bits set to one in an unsigned long long.
 *
 * This is the classical popcount function.
 * Since 2007, it is part of the instruction set of some modern CPU,
 * (>= barcelona for AMD or >= nelhacem for Intel). Alternatively,
 * a fast SWAR algorithm, adding bits in parallel is provided here.
 * This function is massively used to count discs on the board,
 * the mobility, or the stability.
 *
 * @param b 64-bit integer to count bits of.
 * @return the number of bits set.
 */

#ifndef POPCOUNT
int bit_count(unsigned long long b)
{
	int	c;
	#if 0 // defined(USE_GAS_MMX) || defined(USE_MSVC_X86)
	static const unsigned long long M55 = 0x5555555555555555ULL;
	static const unsigned long long M33 = 0x3333333333333333ULL;
	static const unsigned long long M0F = 0x0F0F0F0F0F0F0F0FULL;
	#endif

// MMX does not help much here :-(
	#if 0 // def USE_MSVC_X86
	__m64	m;

	if (hasSSE2) {
		m = *(__m64 *) &b;
		m = _m_psubd(m, _m_pand(_m_psrlqi(m, 1), *(__m64 *) &M55));
		m = _m_paddd(_m_pand(m, *(__m64 *) &M33), _m_pand(_m_psrlqi(m, 2), *(__m64 *) &M33));
		m = _m_pand(_m_paddd(m, _m_psrlqi(m, 4)), *(__m64 *) &M0F);
		c = _m_to_int(_m_psadbw(m, _mm_setzero_si64()));
		_mm_empty();

		return c;
	}

	#elif 0 // defined(USE_GAS_MMX)

	if (hasSSE2) {
		__asm__(
		#ifdef __x86_64__
 			"movq  %1, %%mm1\n\t"
		#else
	 		"movd  %1, %%mm1\n\t"		// to utilize store to load forwarding
			"punpckldq %5, %%mm1\n\t"
		#endif
			"movq  %%mm1, %%mm0\n\t"
			"psrlq $1, %%mm1\n\t"
			"pand  %2, %%mm1\n\t"
			"psubd %%mm1, %%mm0\n\t"

			"movq  %3, %%mm2\n\t"
			"movq  %%mm0, %%mm1\n\t"
			"psrlq $2, %%mm0\n\t"
			"pand  %%mm2, %%mm1\n\t"
			"pand  %%mm2, %%mm0\n\t"
			"paddd %%mm1, %%mm0\n\t"

			"movq  %%mm0, %%mm1\n\t"
			"psrlq $4, %%mm0\n\t"
			"paddd %%mm1, %%mm0\n\t"
			"pand  %4, %%mm0\n\t"

			"pxor  %%mm2, %%mm2\n\t"
			"psadbw %%mm2, %%mm0\n\t"	// SSE2
			"movd	%%mm0, %0\n\t"
			"emms"
		: "=a" (c)
		: "rm" (b), "m" (M55), "m" (M33), "m" (M0F), "m" (((unsigned int *) &b)[1]));

		return c;
	}

	#endif

	b  = b - ((b >> 1) & 0x5555555555555555ULL);
	b  = ((b >> 2) & 0x3333333333333333ULL) + (b & 0x3333333333333333ULL);
#ifdef HAS_CPU_64
	b = (b + (b >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
	c = (b * 0x0101010101010101ULL) >> 56;
#else
	c = (b >> 32) + b;
	c = (c & 0x0F0F0F0F) + ((c >> 4) & 0x0F0F0F0F);
	c = (c * 0x01010101) >> 24;
#endif
	return c;
}
#endif

/**
 * @brief count the number of discs, counting the corners twice.
 *
 * This is a variation of the above algorithm to count the mobility and favour
 * the corners. This function is usefull for move sorting.
 *
 * @param v 64-bit integer to count bits of.
 * @return the number of bit set, counting the corners twice.
 */
int bit_weighted_count(unsigned long long v)
{
#if defined(POPCOUNT)

	return bit_count(v) + bit_count(v & 0x8100000000000081ULL);

#else
	int	c;

	v  = v - ((v >> 1) & 0x1555555555555515ULL) + (v & 0x0100000000000001ULL);
	v  = ((v >> 2) & 0x3333333333333333ULL) + (v & 0x3333333333333333ULL);
#ifdef HAS_CPU_64
	v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
	c = (v * 0x0101010101010101ULL) >> 56;
#else
	c = (v >> 32) + v;
	c = (c & 0x0F0F0F0F) + ((c >> 4) & 0x0F0F0F0F);
	c = (c * 0x01010101) >> 24;
#endif
	return c;
#endif
}

/**
 *
 * @brief Search the first bit set.
 *
 * On CPU with AMD64 or EMT64 instructions, a fast asm
 * code is provided. Alternatively, a fast algorithm based on
 * magic numbers is provided.
 *
 * @param b 64-bit integer.
 * @return the index of the first bit set.
 */
#if !defined(first_bit_32) && !defined(HAS_CPU_64)
int first_bit_32(unsigned int b)
{
#if defined(_MSC_VER)
	unsigned long index;
	_BitScanForward(&index, b);
	return (int) index;

#elif defined(USE_MSVC_X86)
	__asm {
		bsf	eax, word ptr b
	}

#elif defined(USE_GCC_ARM)
	return  __builtin_clz(b & -b) ^ 31;

#else
	static const unsigned char magic[32] = {
		0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
		31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
	};

	return magic[((b & (-b)) * 0x077CB531U) >> 27];
#endif
}
#endif // first_bit_32

#ifndef first_bit
int first_bit(unsigned long long b)
{
#if defined(USE_GAS_X64)
	__asm__("bsfq	%1, %0" : "=r" (b) : "rm" (b));
	return (int) b;

#elif defined(USE_GAS_X86)
	int 	x;
	__asm__ ("bsf	%2, %0\n\t"
		"jnz	1f\n\t"
		"bsf	%1, %0\n\t"
		"addl	$32, %0\n"
	"1:" : "=&q" (x) : "g" ((int) (b >> 32)), "g" ((int) b));
	return x;

#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM))
	unsigned long index;
	_BitScanForward64(&index, b);
	return (int) index;

#elif defined(USE_MSVC_X86)
	__asm {
		bsf	eax, dword ptr b
		jnz	l1
		bsf	eax, dword ptr b+4
		add	eax, 32
	l1:
	}

#elif defined(HAS_CPU_64)
	static const unsigned char magic[64] = {
		63, 0, 58, 1, 59, 47, 53, 2,
		60, 39, 48, 27, 54, 33, 42, 3,
		61, 51, 37, 40, 49, 18, 28, 20,
		55, 30, 34, 11, 43, 14, 22, 4,
		62, 57, 46, 52, 38, 26, 32, 41,
		50, 36, 17, 19, 29, 10, 13, 21,
		56, 45, 25, 31, 35, 16, 9, 12,
		44, 24, 15, 8, 23, 7, 6, 5
	};

	return magic[((b & (-b)) * 0x07EDD5E59A4E28C2ULL) >> 58];

#else
	const unsigned int lb = (unsigned int) b;
	if (lb) {
		return first_bit_32(lb);
	} else {
		return 32 + first_bit_32(b >> 32);
	}
#endif
}
#endif // first_bit

#if 0
/**
 * @brief Search the next bit set.
 *
 * In practice, clear the first bit set and search the next one.
 *
 * @param b 64-bit integer.
  * @return the index of the next bit set.
 */
int next_bit(unsigned long long *b)
{
	*b &= *b - 1;
	return first_bit(*b);
}
#endif

#ifndef last_bit
/**
 * @brief Search the last bit set (same as log2()).
 *
 * On CPU with AMD64 or EMT64 instructions, a fast asm
 * code is provided. Alternatively, a fast algorithm based on
 * magic numbers is provided.
 *
 * @param b 64-bit integer.
 * @return the index of the last bit set.
 */
int last_bit(unsigned long long b)
{
#if defined(USE_GAS_X64)
	__asm__("bsrq	%1, %0" :"=r" (b) :"rm" (b));
	return b;

#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM))
	unsigned long index;
	_BitScanReverse64(&index, b);
	return (int) index;

#elif defined(USE_GAS_X86)
	int	x;
	__asm__ ("bsr	%1, %0\n\t"
		"leal	32(%0), %0\n\t"
		"jnz	1f\n\t"
		"bsr	%2, %0\n\t"
        "1:" : "=&q" (x) : "g" ((int) (b >> 32)), "g" ((int) b));
	return x;

#elif defined(USE_GCC_ARM)
	const unsigned int hb = b >> 32;
	if (hb) {
		return 63 - __builtin_clz(hb);
	} else {
		return 31 - __builtin_clz((int) b);
	}

#elif defined(USE_MSVC_X86)
	__asm {
		bsr	eax, dword ptr b+4
		lea	eax, [eax+32]
		jnz	l1
		bsr	eax, dword ptr b
	l1:
	}

#elif defined(HAS_CPU_64)
	static const unsigned char magic[64] = {
		63, 0, 58, 1, 59, 47, 53, 2,
		60, 39, 48, 27, 54, 33, 42, 3,
		61, 51, 37, 40, 49, 18, 28, 20,
		55, 30, 34, 11, 43, 14, 22, 4,
		62, 57, 46, 52, 38, 26, 32, 41,
		50, 36, 17, 19, 29, 10, 13, 21,
		56, 45, 25, 31, 35, 16, 9, 12,
		44, 24, 15, 8, 23, 7, 6, 5
	};

	b |= b >> 1;
	b |= b >> 2;
	b |= b >> 4;
	b |= b >> 8;
	b |= b >> 16;
	b |= b >> 32;
	b = (b >> 1) + 1;

	return magic[(b * 0x07EDD5E59A4E28C2ULL) >> 58];

#else
	static const unsigned char clz_table_4bit[16] = { 4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
	int	n = 63;
	unsigned int	x;

	x = b >> 32;
	if (x == 0) { n = 31; x = (unsigned int) b; }
	if ((x & 0xFFFF0000) == 0) { n -= 16; x <<= 16; }
	if ((x & 0xFF000000) == 0) { n -=  8; x <<=  8; }
	if ((x & 0xF0000000) == 0) { n -=  4; x <<=  4; }
	n -= clz_table_4bit[x >> (32 - 4)];
	return n;
#endif
}
#endif // last_bit

#ifndef bswap_short
/**
 * @brief Swap bytes of a short (little <-> big endian).
 * @param s An unsigned short.
 * @return The mirrored short.
 */
unsigned short bswap_short(unsigned short s)
{
	return (unsigned short) ((s >> 8) & 0x00FF) | ((s & 0x00FF) <<  8);
}
#endif

#ifndef bswap_int
/**
 * @brief Mirror the unsigned int (little <-> big endian).
 * @param i An unsigned int.
 * @return The mirrored int.
 */
unsigned int bswap_int(unsigned int i)
{
	i = ((i >>  8) & 0x00FF00FFU) | ((i & 0x00FF00FFU) <<  8);
	i = (i >> 16) | (i << 16);
	return i;
}

/**
 * @brief Mirror the unsigned long long (exchange the lines A - H, B - G, C - F & D - E.).
 * @param b An unsigned long long
 * @return The mirrored unsigned long long.
 */
unsigned long long vertical_mirror(unsigned long long b)
{
	b = ((b >>  8) & 0x00FF00FF00FF00FFULL) | ((b & 0x00FF00FF00FF00FFULL) <<  8);
	b = ((b >> 16) & 0x0000FFFF0000FFFFULL) | ((b & 0x0000FFFF0000FFFFULL) << 16);
	b = (b >> 32) | (b << 32);
	return b;
}
#endif // bswap_int

/**
 * @brief Mirror the unsigned long long (exchange the line 1 - 8, 2 - 7, 3 - 6 & 4 - 5).
 * @param b An unsigned long long.
 * @return The mirrored unsigned long long.
 */
unsigned int horizontal_mirror_32(unsigned int b)
{
	b = ((b >> 1) & 0x55555555U) +  2 * (b & 0x55555555U);
	b = ((b >> 2) & 0x33333333U) +  4 * (b & 0x33333333U);
	b = ((b >> 4) & 0x0F0F0F0FU) + 16 * (b & 0x0F0F0F0FU);
	return b;
}

unsigned long long horizontal_mirror(unsigned long long b)
{
#ifdef HAS_CPU_64
	b = ((b >> 1) & 0x5555555555555555ULL) | ((b & 0x5555555555555555ULL) << 1);
	b = ((b >> 2) & 0x3333333333333333ULL) | ((b & 0x3333333333333333ULL) << 2);
	b = ((b >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((b & 0x0F0F0F0F0F0F0F0FULL) << 4);
	return b;
#else
	return ((unsigned long long) horizontal_mirror_32(b >> 32) << 32)
		| horizontal_mirror_32((unsigned int) b);
#endif
}

/**
 * @brief Transpose the unsigned long long (symetry % A1-H8 diagonal, or swap axes).
 * @param b An unsigned long long
 * @return The transposed unsigned long long.
 */
#ifdef __AVX2__
unsigned long long transpose(unsigned long long b)
{
	static const V4DI s3210 = {{ 3, 2, 1, 0 }};
	__m256i	v = _mm256_sllv_epi64(_mm256_broadcastq_epi64(_mm_cvtsi64_si128(b)), s3210.v4);
	return ((unsigned long long) _mm256_movemask_epi8(v) << 32)
		| (unsigned int) _mm256_movemask_epi8(_mm256_slli_epi64(v, 4));
}

#else
unsigned long long transpose(unsigned long long b)
{
	unsigned long long t;

	t = (b ^ (b >> 7)) & 0x00aa00aa00aa00aaULL;
	b = b ^ t ^ (t << 7);
	t = (b ^ (b >> 14)) & 0x0000cccc0000ccccULL;
	b = b ^ t ^ (t << 14);
	t = (b ^ (b >> 28)) & 0x00000000f0f0f0f0ULL;
	b = b ^ t ^ (t << 28);

	return b;
}
#endif // __AVX2__

/**
 * @brief Get a random set bit index.
 *
 * @param b The unsigned long long.
 * @param r The pseudo-number generator.
 * @return a random bit index, or -1 if b value is zero.
 */
int get_rand_bit(unsigned long long b, Random *r)
{
	int n = bit_count(b), x;

	if (n == 0) return -1;

	n = random_get(r) % n;
	foreach_bit(x, b) if (n-- == 0) return x;

	return -2; // impossible.
}

/**
 * @brief Print an unsigned long long as a board.
 *
 * Write a 64-bit long number as an Othello board.
 *
 * @param b The unsigned long long.
 * @param f Output stream.
 */
void bitboard_write(unsigned long long b, FILE *f)
{
	int i, j;
	static const char color[2] = ".X";

	fputs("  A B C D E F G H\n", f);
	for (i = 0; i < 8; ++i) {
		fputc(i + '1', f);
		fputc(' ', f);
		for (j = 0; j < 8; ++j) {
			fputc(color[b & 1], f);
			fputc(' ', f);
			b >>= 1;
		}
		fputc(i + '1', f);
		fputc('\n', f);
	}
	fputs("  A B C D E F G H\n", f);
}
