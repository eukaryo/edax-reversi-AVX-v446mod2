/**
 * @file eval_sse.c
 *
 * SSE/AVX translation of some eval.c functions
 *
 * @date 2018
 * @author Toshihiko Okuhara
 * @version 4.4
 */

#include <assert.h>

#include "bit.h"
#include "board.h"
#include "move.h"
#include "eval.h"

typedef union {
	unsigned short us[48];
#if defined(hasSSE2) || defined(USE_MSVC_X86)
	__m128i	v8[6];
#endif
#ifdef __AVX2__
	__m256i	v16[3];
#endif
}
#if defined(__GNUC__) && !defined(hasSSE2)
__attribute__ ((aligned (16)))
#endif
EVAL_FEATURE_V;

static const EVAL_FEATURE_V EVAL_FEATURE[65] = {
	{{ // a1
		 6561,     0,     0,     0,   243,     0,     0,     0,  6561,     0,  6561,     0, 19683,     0, 19683,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  2187,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // b1
		 2187,     0,     0,     0,    27,     0,     0,     0,  2187,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,  2187,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   729,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // c1
		   81,     0,     0,     0,     9,     0,     0,     0,   729,     0,     0,     0,  6561,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,  2187,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,   243,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // d1
		    0,     0,     0,     0,     3,     1,     0,     0,   243,     0,     0,     0,  2187,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  2187,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,    81,     0,     0,     0,    27,     0,     0,     0,     0,     0
	}}, {{ // e1
		    0,     0,     0,     0,     1,     3,     0,     0,    81,     0,     0,     0,     9,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  2187,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,    81,     0,     0,     0,    27,     0,     0,     0
	}}, {{ // f1
		    0,    81,     0,     0,     0,     9,     0,     0,    27,     0,     0,     0,     3,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,  2187,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,   243,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // g1
		    0,  2187,     0,     0,     0,    27,     0,     0,     9,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,  2187,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,   729,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // h1
		    0,  6561,     0,     0,     0,   243,     0,     0,     3,     0,     0,  6561,     1,     0,     0, 19683,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // a2
		  729,     0,     0,     0,   729,     0,     0,     0,     0,     0,  2187,     0,     0,     0,     0,     0,
		 2187,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		  729,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // b2
		  243,     0,     0,     0,    81,     0,     0,     0, 19683,     0, 19683,     0,     0,     0,     0,     0,
		  729,     0,   729,     0,     0,     0,     0,     0,     0,     0,     0,     0,   729,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // c2
		    9,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   729,     0,     0,     0,
		  243,     0,     0,     0,     0,     0,   729,     0,     0,     0,     0,     0,     0,     0,   243,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     9,     0,     0,     0,     0,     0
	}}, {{ // d2
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   243,     0,     0,     0,
		   81,     0,     0,     0,     0,     0,     0,     0,     0,     0,   729,     0,     0,     0,     0,     0,
		    0,     0,    81,     0,     0,     0,     0,     0,    27,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // e2
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    81,     0,     0,     0,
		   27,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   729,     0,     0,     0,     0,
		    0,     0,     0,     0,    81,     0,    27,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // f2
		    0,     9,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    27,     0,     0,     0,
		    9,     0,     0,     0,     0,     0,     0,   729,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,   243,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     9,     0,     0,     0
	}}, {{ // g2
		    0,   243,     0,     0,     0,    81,     0,     0,     1,     0,     0, 19683,     0,     0,     0,     0,
		    3,     0,     0,   729,     0,     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // h2
		    0,   729,     0,     0,     0,   729,     0,     0,     0,     0,     0,  2187,     0,     0,     0,     0,
		    1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   729,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // a3
		   27,     0,     0,     0,  2187,     0,     0,     0,     0,     0,   729,     0,     0,     0,  6561,     0,
		    0,     0,     0,     0,  2187,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,   243,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // b3
		    3,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   729,     0,
		    0,     0,   243,     0,   729,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		  243,     0,     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,     0,     0,     0,     0
	}}, {{ // c3
		    1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,   243,     0,   243,     0,     0,     0,     0,     0,   243,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     9,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // d3
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,    81,     0,     0,     0,     0,     0,   243,     0,     0,     0,    81,     0,
		    0,     0,     0,     0,    27,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // e3
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,    27,     0,     0,     0,     0,     0,     0,   243,     0,     0,     0,     0,
		    0,    81,    27,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // f3
		    0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     9,     0,     0,   243,     0,     0,     0,     0,     0,     9,     0,     0,
		    0,     0,     0,     0,     0,     0,     9,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // g3
		    0,     3,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   729,
		    0,     0,     0,   243,     3,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   243,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,     0,     0
	}}, {{ // h3
		    0,    27,     0,     0,     0,  2187,     0,     0,     0,     0,     0,   729,     0,     0,     0,  6561,
		    0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,   243,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // a4
		    0,     0,     0,     0,  6561,     0, 19683,     0,     0,     0,   243,     0,     0,     0,  2187,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,  2187,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,    81,     0,     0,     1,     0,     0,     0,     0,     0
	}}, {{ // b4
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   243,     0,
		    0,     0,    81,     0,     0,     0,     0,     0,   729,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,    81,     0,     0,     0,     0,     3,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // c4
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,    81,     0,   243,     0,     0,     0,     0,     0,     0,     0,
		   81,     0,     0,     0,     9,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // d4
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,    81,     0,    81,     0,    81,     0,     0,     0,
		    0,    27,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // e4
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,    27,     0,     0,    81,     0,    27,    27,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // f4
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,    81,     9,     0,     0,     0,     0,     0,     0,    81,
		    0,     0,     9,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // g4
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   243,
		    0,     0,     0,    81,     0,     0,     0,     0,     3,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,    81,     3,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // h4
		    0,     0,     0,     0,     0,  6561,     0, 19683,     0,     0,     0,   243,     0,     0,     0,  2187,
		    0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,    81,     0,     0,     1,     0,     0,     0
	}}, {{ // a5
		    0,     0,     0,     0, 19683,     0,  6561,     0,     0,     0,    81,     0,     0,     0,     9,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,  2187,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,    27,     0,     0,     0,     0
	}}, {{ // b5
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    81,     0,
		    0,     0,    27,     0,     0,     0,     0,     0,     0,   729,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     3,     0,     0,    27,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // c5
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,    27,     0,     0,   243,     0,     0,     0,     0,     0,     0,
		    0,     9,     0,    27,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // d5
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,    81,    27,     0,     0,    81,     0,     0,
		   27,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // e5
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,    27,     0,    27,    27,     0,     0,    27,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // f5
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,    27,     0,     9,     0,     0,     0,     0,     9,     0,
		    0,     0,     0,     0,     0,    27,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // g5
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    81,
		    0,     0,     0,    27,     0,     0,     0,     0,     0,     3,     0,     0,     0,     0,     0,     0,
		    0,     0,     3,     0,     0,     0,     0,     0,     0,    27,     0,     0,     0,     0,     0,     0
	}}, {{ // h5
		    0,     0,     0,     0,     0, 19683,     0,  6561,     0,     0,     0,    81,     0,     0,     0,     9,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,    27,     0,     0
	}}, {{ // a6
		    0,     0,    81,     0,     0,     0,  2187,     0,     0,     0,    27,     0,     0,     0,     3,     0,
		    0,     0,     0,     0,     0,  2187,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // b6
		    0,     0,     9,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    27,     0,
		    0,     0,     9,     0,     0,   729,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     3,     0,     0,     0,     0,     0,     0,     0,     0,     0,     9,     0,     0,     0,     0
	}}, {{ // c6
		    0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,   243,     9,     0,     0,     0,     0,     0,     0,   243,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     9,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // d6
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,    81,     0,     0,     0,     0,     9,     0,     0,     0,     0,     9,
		    0,     0,     0,     9,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // e6
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,    27,     0,     0,     0,     0,     0,     9,     0,     0,     0,     0,
		    9,     0,     0,     0,     0,     9,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // f6
		    0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     9,     0,     9,     0,     0,     0,     0,     9,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     9,     0,     0,     0,     0,     0,     0
	}}, {{ // g6
		    0,     0,     0,     9,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    27,
		    0,     0,     0,     9,     0,     3,     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     9,     0,     0
	}}, {{ // h6
		    0,     0,     0,    81,     0,     0,     0,  2187,     0,     0,     0,    27,     0,     0,     0,     3,
		    0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // a7
		    0,     0,  2187,     0,     0,     0,   729,     0,     0,     0,     9,     0,     0,     0,     0,     0,
		    0,  2187,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // b7
		    0,     0,   243,     0,     0,     0,    81,     0,     0, 19683,     1,     0,     0,     0,     0,     0,
		    0,   729,     3,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   729,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // c7
		    0,     0,     3,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   729,     0,     0,
		    0,   243,     0,     0,     0,     0,     3,     0,     0,     0,     0,     0,     0,     0,     0,     3,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,     0,     0,     0
	}}, {{ // d7
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,   243,     0,     0,
		    0,    81,     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     3,     0,     3,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // e7
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,    81,     0,     0,
		    0,    27,     0,     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,     0,     0,     0,
		    0,     0,     0,     3,     0,     0,     0,     0,     0,     3,     0,     0,     0,     0,     0,     0
	}}, {{ // f7
		    0,     0,     0,     3,     0,     0,     0,     0,     0,     0,     0,     0,     0,    27,     0,     0,
		    0,     9,     0,     0,     0,     0,     0,     3,     0,     0,     0,     0,     0,     0,     0,     0,
		    3,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,     0
	}}, {{ // g7
		    0,     0,     0,   243,     0,     0,     0,    81,     0,     1,     0,     1,     0,     0,     0,     0,
		    0,     3,     0,     3,     0,     0,     0,     0,     0,     0,     0,     0,     3,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // h7
		    0,     0,     0,  2187,     0,     0,     0,   729,     0,     0,     0,     9,     0,     0,     0,     0,
		    0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // a8
		    0,     0,  6561,     0,     0,     0,   243,     0,     0,  6561,     3,     0,     0, 19683,     1,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,  2187,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // b8
		    0,     0,   729,     0,     0,     0,    27,     0,     0,  2187,     0,     0,     0,     0,     0,     0,
		    0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // c8
		    0,     0,    27,     0,     0,     0,     9,     0,     0,   729,     0,     0,     0,  6561,     0,     0,
		    0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // d8
		    0,     0,     0,     0,     0,     0,     3,     1,     0,   243,     0,     0,     0,  2187,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     1,     0,     0,     0,     0
	}}, {{ // e8
		    0,     0,     0,     0,     0,     0,     1,     3,     0,    81,     0,     0,     0,     9,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     1,     0,     0
	}}, {{ // f8
		    0,     0,     0,    27,     0,     0,     0,     9,     0,    27,     0,     0,     0,     3,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // g8
		    0,     0,     0,   729,     0,     0,     0,    27,     0,     9,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    1,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // h8
		    0,     0,     0,  6561,     0,     0,     0,   243,     0,     3,     0,     3,     0,     1,     0,     1,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}, {{ // PASS
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
		    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0
	}}
};

#if defined(hasSSE2) || defined(USE_MSVC_X86)

static void eval_update_sse_0(Eval *eval, const Move *move)
{
	int	x = move->x;
	unsigned long long f = move->flipped;
#ifdef __AVX2__
	__m256i	f0 = _mm256_sub_epi16(eval->feature.v16[0], _mm256_slli_epi16(EVAL_FEATURE[x].v16[0], 1));
	__m256i	f1 = _mm256_sub_epi16(eval->feature.v16[1], _mm256_slli_epi16(EVAL_FEATURE[x].v16[1], 1));
	__m256i	f2 = _mm256_sub_epi16(eval->feature.v16[2], _mm256_slli_epi16(EVAL_FEATURE[x].v16[2], 1));

	foreach_bit(x, f) {
		f0 = _mm256_sub_epi16(f0, EVAL_FEATURE[x].v16[0]);
		f1 = _mm256_sub_epi16(f1, EVAL_FEATURE[x].v16[1]);
		f2 = _mm256_sub_epi16(f2, EVAL_FEATURE[x].v16[2]);
	}
	eval->feature.v16[0] = f0;
	eval->feature.v16[1] = f1;
	eval->feature.v16[2] = f2;

#else
	__m128i	f0 = _mm_sub_epi16(eval->feature.v8[0], _mm_slli_epi16(EVAL_FEATURE[x].v8[0], 1));
	__m128i	f1 = _mm_sub_epi16(eval->feature.v8[1], _mm_slli_epi16(EVAL_FEATURE[x].v8[1], 1));
	__m128i	f2 = _mm_sub_epi16(eval->feature.v8[2], _mm_slli_epi16(EVAL_FEATURE[x].v8[2], 1));
	__m128i	f3 = _mm_sub_epi16(eval->feature.v8[3], _mm_slli_epi16(EVAL_FEATURE[x].v8[3], 1));
	__m128i	f4 = _mm_sub_epi16(eval->feature.v8[4], _mm_slli_epi16(EVAL_FEATURE[x].v8[4], 1));
	__m128i	f5 = _mm_sub_epi16(eval->feature.v8[5], _mm_slli_epi16(EVAL_FEATURE[x].v8[5], 1));

#ifdef HAS_CPU_64
	foreach_bit(x, f) {
		f0 = _mm_sub_epi16(f0, EVAL_FEATURE[x].v8[0]);
		f1 = _mm_sub_epi16(f1, EVAL_FEATURE[x].v8[1]);
		f2 = _mm_sub_epi16(f2, EVAL_FEATURE[x].v8[2]);
		f3 = _mm_sub_epi16(f3, EVAL_FEATURE[x].v8[3]);
		f4 = _mm_sub_epi16(f4, EVAL_FEATURE[x].v8[4]);
		f5 = _mm_sub_epi16(f5, EVAL_FEATURE[x].v8[5]);
	}

#else
	unsigned int	fl = (unsigned int) f;
	unsigned int	fh = (unsigned int) (f >> 32);

	foreach_bit_32(x, fl) {
		f0 = _mm_sub_epi16(f0, EVAL_FEATURE[x].v8[0]);
		f1 = _mm_sub_epi16(f1, EVAL_FEATURE[x].v8[1]);
		f2 = _mm_sub_epi16(f2, EVAL_FEATURE[x].v8[2]);
		f3 = _mm_sub_epi16(f3, EVAL_FEATURE[x].v8[3]);
		f4 = _mm_sub_epi16(f4, EVAL_FEATURE[x].v8[4]);
		f5 = _mm_sub_epi16(f5, EVAL_FEATURE[x].v8[5]);
	}
	foreach_bit_32(x, fh) {
		f0 = _mm_sub_epi16(f0, EVAL_FEATURE[x + 32].v8[0]);
		f1 = _mm_sub_epi16(f1, EVAL_FEATURE[x + 32].v8[1]);
		f2 = _mm_sub_epi16(f2, EVAL_FEATURE[x + 32].v8[2]);
		f3 = _mm_sub_epi16(f3, EVAL_FEATURE[x + 32].v8[3]);
		f4 = _mm_sub_epi16(f4, EVAL_FEATURE[x + 32].v8[4]);
		f5 = _mm_sub_epi16(f5, EVAL_FEATURE[x + 32].v8[5]);
	}
#endif
	eval->feature.v8[0] = f0;
	eval->feature.v8[1] = f1;
	eval->feature.v8[2] = f2;
	eval->feature.v8[3] = f3;
	eval->feature.v8[4] = f4;
	eval->feature.v8[5] = f5;
#endif
}

/**
 * @brief Update the features after a player's move.
 *
 * @param eval  Evaluation function.
 * @param move  Move.
 */
static void eval_update_sse_1(Eval *eval, const Move *move)
{
	int	x = move->x;
	unsigned long long f = move->flipped;
#ifdef __AVX2__
	__m256i	f0 = _mm256_sub_epi16(eval->feature.v16[0], EVAL_FEATURE[x].v16[0]);
	__m256i	f1 = _mm256_sub_epi16(eval->feature.v16[1], EVAL_FEATURE[x].v16[1]);
	__m256i	f2 = _mm256_sub_epi16(eval->feature.v16[2], EVAL_FEATURE[x].v16[2]);

	foreach_bit(x, f) {
		f0 = _mm256_add_epi16(f0, EVAL_FEATURE[x].v16[0]);
		f1 = _mm256_add_epi16(f1, EVAL_FEATURE[x].v16[1]);
		f2 = _mm256_add_epi16(f2, EVAL_FEATURE[x].v16[2]);
	}
	eval->feature.v16[0] = f0;
	eval->feature.v16[1] = f1;
	eval->feature.v16[2] = f2;

#else
	__m128i	f0 = _mm_sub_epi16(eval->feature.v8[0], EVAL_FEATURE[x].v8[0]);
	__m128i	f1 = _mm_sub_epi16(eval->feature.v8[1], EVAL_FEATURE[x].v8[1]);
	__m128i	f2 = _mm_sub_epi16(eval->feature.v8[2], EVAL_FEATURE[x].v8[2]);
	__m128i	f3 = _mm_sub_epi16(eval->feature.v8[3], EVAL_FEATURE[x].v8[3]);
	__m128i	f4 = _mm_sub_epi16(eval->feature.v8[4], EVAL_FEATURE[x].v8[4]);
	__m128i	f5 = _mm_sub_epi16(eval->feature.v8[5], EVAL_FEATURE[x].v8[5]);

#ifdef HAS_CPU_64
	foreach_bit(x, f) {
		f0 = _mm_add_epi16(f0, EVAL_FEATURE[x].v8[0]);
		f1 = _mm_add_epi16(f1, EVAL_FEATURE[x].v8[1]);
		f2 = _mm_add_epi16(f2, EVAL_FEATURE[x].v8[2]);
		f3 = _mm_add_epi16(f3, EVAL_FEATURE[x].v8[3]);
		f4 = _mm_add_epi16(f4, EVAL_FEATURE[x].v8[4]);
		f5 = _mm_add_epi16(f5, EVAL_FEATURE[x].v8[5]);
	}

#else
	unsigned int	fl = (unsigned int) f;
	unsigned int	fh = (unsigned int) (f >> 32);

	foreach_bit_32(x, fl) {
		f0 = _mm_add_epi16(f0, EVAL_FEATURE[x].v8[0]);
		f1 = _mm_add_epi16(f1, EVAL_FEATURE[x].v8[1]);
		f2 = _mm_add_epi16(f2, EVAL_FEATURE[x].v8[2]);
		f3 = _mm_add_epi16(f3, EVAL_FEATURE[x].v8[3]);
		f4 = _mm_add_epi16(f4, EVAL_FEATURE[x].v8[4]);
		f5 = _mm_add_epi16(f5, EVAL_FEATURE[x].v8[5]);
	}
	foreach_bit_32(x, fh) {
		f0 = _mm_add_epi16(f0, EVAL_FEATURE[x + 32].v8[0]);
		f1 = _mm_add_epi16(f1, EVAL_FEATURE[x + 32].v8[1]);
		f2 = _mm_add_epi16(f2, EVAL_FEATURE[x + 32].v8[2]);
		f3 = _mm_add_epi16(f3, EVAL_FEATURE[x + 32].v8[3]);
		f4 = _mm_add_epi16(f4, EVAL_FEATURE[x + 32].v8[4]);
		f5 = _mm_add_epi16(f5, EVAL_FEATURE[x + 32].v8[5]);
	}

#endif
	eval->feature.v8[0] = f0;
	eval->feature.v8[1] = f1;
	eval->feature.v8[2] = f2;
	eval->feature.v8[3] = f3;
	eval->feature.v8[4] = f4;
	eval->feature.v8[5] = f5;
#endif
}

static void eval_restore_sse_0(Eval *eval, const Move *move)
{
	int	x = move->x;
	unsigned long long f = move->flipped;
#ifdef __AVX2__
	__m256i	f0 = _mm256_add_epi16(eval->feature.v16[0], _mm256_slli_epi16(EVAL_FEATURE[x].v16[0], 1));
	__m256i	f1 = _mm256_add_epi16(eval->feature.v16[1], _mm256_slli_epi16(EVAL_FEATURE[x].v16[1], 1));
	__m256i	f2 = _mm256_add_epi16(eval->feature.v16[2], _mm256_slli_epi16(EVAL_FEATURE[x].v16[2], 1));

	foreach_bit(x, f) {
		f0 = _mm256_add_epi16(f0, EVAL_FEATURE[x].v16[0]);
		f1 = _mm256_add_epi16(f1, EVAL_FEATURE[x].v16[1]);
		f2 = _mm256_add_epi16(f2, EVAL_FEATURE[x].v16[2]);
	}
	eval->feature.v16[0] = f0;
	eval->feature.v16[1] = f1;
	eval->feature.v16[2] = f2;

#else
	__m128i	f0 = _mm_add_epi16(eval->feature.v8[0], _mm_slli_epi16(EVAL_FEATURE[x].v8[0], 1));
	__m128i	f1 = _mm_add_epi16(eval->feature.v8[1], _mm_slli_epi16(EVAL_FEATURE[x].v8[1], 1));
	__m128i	f2 = _mm_add_epi16(eval->feature.v8[2], _mm_slli_epi16(EVAL_FEATURE[x].v8[2], 1));
	__m128i	f3 = _mm_add_epi16(eval->feature.v8[3], _mm_slli_epi16(EVAL_FEATURE[x].v8[3], 1));
	__m128i	f4 = _mm_add_epi16(eval->feature.v8[4], _mm_slli_epi16(EVAL_FEATURE[x].v8[4], 1));
	__m128i	f5 = _mm_add_epi16(eval->feature.v8[5], _mm_slli_epi16(EVAL_FEATURE[x].v8[5], 1));

#ifdef HAS_CPU_64
	foreach_bit(x, f) {
		f0 = _mm_add_epi16(f0, EVAL_FEATURE[x].v8[0]);
		f1 = _mm_add_epi16(f1, EVAL_FEATURE[x].v8[1]);
		f2 = _mm_add_epi16(f2, EVAL_FEATURE[x].v8[2]);
		f3 = _mm_add_epi16(f3, EVAL_FEATURE[x].v8[3]);
		f4 = _mm_add_epi16(f4, EVAL_FEATURE[x].v8[4]);
		f5 = _mm_add_epi16(f5, EVAL_FEATURE[x].v8[5]);
	}

#else
	unsigned int	fl = (unsigned int) f;
	unsigned int	fh = (unsigned int) (f >> 32);

	foreach_bit_32(x, fl) {
		f0 = _mm_add_epi16(f0, EVAL_FEATURE[x].v8[0]);
		f1 = _mm_add_epi16(f1, EVAL_FEATURE[x].v8[1]);
		f2 = _mm_add_epi16(f2, EVAL_FEATURE[x].v8[2]);
		f3 = _mm_add_epi16(f3, EVAL_FEATURE[x].v8[3]);
		f4 = _mm_add_epi16(f4, EVAL_FEATURE[x].v8[4]);
		f5 = _mm_add_epi16(f5, EVAL_FEATURE[x].v8[5]);
	}
	foreach_bit_32(x, fh) {
		f0 = _mm_add_epi16(f0, EVAL_FEATURE[x + 32].v8[0]);
		f1 = _mm_add_epi16(f1, EVAL_FEATURE[x + 32].v8[1]);
		f2 = _mm_add_epi16(f2, EVAL_FEATURE[x + 32].v8[2]);
		f3 = _mm_add_epi16(f3, EVAL_FEATURE[x + 32].v8[3]);
		f4 = _mm_add_epi16(f4, EVAL_FEATURE[x + 32].v8[4]);
		f5 = _mm_add_epi16(f5, EVAL_FEATURE[x + 32].v8[5]);
	}
#endif

	eval->feature.v8[0] = f0;
	eval->feature.v8[1] = f1;
	eval->feature.v8[2] = f2;
	eval->feature.v8[3] = f3;
	eval->feature.v8[4] = f4;
	eval->feature.v8[5] = f5;
#endif
}

static void eval_restore_sse_1(Eval *eval, const Move *move)
{
	int	x = move->x;
	unsigned long long f = move->flipped;
#ifdef __AVX2__
	__m256i	f0 = _mm256_add_epi16(eval->feature.v16[0], EVAL_FEATURE[x].v16[0]);
	__m256i	f1 = _mm256_add_epi16(eval->feature.v16[1], EVAL_FEATURE[x].v16[1]);
	__m256i	f2 = _mm256_add_epi16(eval->feature.v16[2], EVAL_FEATURE[x].v16[2]);

	foreach_bit(x, f) {
		f0 = _mm256_sub_epi16(f0, EVAL_FEATURE[x].v16[0]);
		f1 = _mm256_sub_epi16(f1, EVAL_FEATURE[x].v16[1]);
		f2 = _mm256_sub_epi16(f2, EVAL_FEATURE[x].v16[2]);
	}
	eval->feature.v16[0] = f0;
	eval->feature.v16[1] = f1;
	eval->feature.v16[2] = f2;

#else
	__m128i	f0 = _mm_add_epi16(eval->feature.v8[0], EVAL_FEATURE[x].v8[0]);
	__m128i	f1 = _mm_add_epi16(eval->feature.v8[1], EVAL_FEATURE[x].v8[1]);
	__m128i	f2 = _mm_add_epi16(eval->feature.v8[2], EVAL_FEATURE[x].v8[2]);
	__m128i	f3 = _mm_add_epi16(eval->feature.v8[3], EVAL_FEATURE[x].v8[3]);
	__m128i	f4 = _mm_add_epi16(eval->feature.v8[4], EVAL_FEATURE[x].v8[4]);
	__m128i	f5 = _mm_add_epi16(eval->feature.v8[5], EVAL_FEATURE[x].v8[5]);

#ifdef HAS_CPU_64
	foreach_bit(x, f) {
		f0 = _mm_sub_epi16(f0, EVAL_FEATURE[x].v8[0]);
		f1 = _mm_sub_epi16(f1, EVAL_FEATURE[x].v8[1]);
		f2 = _mm_sub_epi16(f2, EVAL_FEATURE[x].v8[2]);
		f3 = _mm_sub_epi16(f3, EVAL_FEATURE[x].v8[3]);
		f4 = _mm_sub_epi16(f4, EVAL_FEATURE[x].v8[4]);
		f5 = _mm_sub_epi16(f5, EVAL_FEATURE[x].v8[5]);
	}

#else
	unsigned int	fl = (unsigned int) f;
	unsigned int	fh = (unsigned int) (f >> 32);

	foreach_bit_32(x, fl) {
		f0 = _mm_sub_epi16(f0, EVAL_FEATURE[x].v8[0]);
		f1 = _mm_sub_epi16(f1, EVAL_FEATURE[x].v8[1]);
		f2 = _mm_sub_epi16(f2, EVAL_FEATURE[x].v8[2]);
		f3 = _mm_sub_epi16(f3, EVAL_FEATURE[x].v8[3]);
		f4 = _mm_sub_epi16(f4, EVAL_FEATURE[x].v8[4]);
		f5 = _mm_sub_epi16(f5, EVAL_FEATURE[x].v8[5]);
	}
	foreach_bit_32(x, fh) {
		f0 = _mm_sub_epi16(f0, EVAL_FEATURE[x + 32].v8[0]);
		f1 = _mm_sub_epi16(f1, EVAL_FEATURE[x + 32].v8[1]);
		f2 = _mm_sub_epi16(f2, EVAL_FEATURE[x + 32].v8[2]);
		f3 = _mm_sub_epi16(f3, EVAL_FEATURE[x + 32].v8[3]);
		f4 = _mm_sub_epi16(f4, EVAL_FEATURE[x + 32].v8[4]);
		f5 = _mm_sub_epi16(f5, EVAL_FEATURE[x + 32].v8[5]);
	}
#endif
	eval->feature.v8[0] = f0;
	eval->feature.v8[1] = f1;
	eval->feature.v8[2] = f2;
	eval->feature.v8[3] = f3;
	eval->feature.v8[4] = f4;
	eval->feature.v8[5] = f5;
#endif
}

#else	// SSE dispatch (Eval may not be aligned)

static void eval_update_sse_0(Eval *eval, const Move *move)
{
	int	x = move->x;
	unsigned int	fl = (unsigned int) move->flipped;
	unsigned int	fh = (unsigned int) (move->flipped >> 32);

	__asm__ (
		"movdqa	%2, %%xmm0\n\t"		"movdqa	%3, %%xmm1\n\t"
		"movdqu	%0, %%xmm2\n\t"		"movdqu	%1, %%xmm3\n\t"
		"psllw	$1, %%xmm0\n\t"		"psllw	$1, %%xmm1\n\t"
		"psubw	%%xmm0, %%xmm2\n\t"	"psubw	%%xmm1, %%xmm3\n"
	: :  "m" (eval->feature.us[0]), "m" (eval->feature.us[8]), "m" (EVAL_FEATURE[x].us[0]),  "m" (EVAL_FEATURE[x].us[8]));
	__asm__ (
		"movdqa	%2, %%xmm0\n\t"		"movdqa	%3, %%xmm1\n\t"
		"movdqu	%0, %%xmm4\n\t"		"movdqu	%1, %%xmm5\n\t"
		"psllw	$1, %%xmm0\n\t"		"psllw	$1, %%xmm1\n\t"
		"psubw	%%xmm0, %%xmm4\n\t"	"psubw	%%xmm1, %%xmm5\n"
	: :  "m" (eval->feature.us[16]), "m" (eval->feature.us[24]), "m" (EVAL_FEATURE[x].us[16]),  "m" (EVAL_FEATURE[x].us[24]));
	__asm__ (
		"movdqa	%2, %%xmm0\n\t"		"movdqa	%3, %%xmm1\n\t"
		"movdqu	%0, %%xmm6\n\t"		"movdqu	%1, %%xmm7\n\t"
		"psllw	$1, %%xmm0\n\t"		"psllw	$1, %%xmm1\n\t"
		"psubw	%%xmm0, %%xmm6\n\t"	"psubw	%%xmm1, %%xmm7\n"
	: :  "m" (eval->feature.us[32]), "m" (eval->feature.us[40]), "m" (EVAL_FEATURE[x].us[32]),  "m" (EVAL_FEATURE[x].us[40]));

	foreach_bit_32(x, fl) {
		__asm__ (
			"psubw	%0, %%xmm2\n\t"
			"psubw	%1, %%xmm3\n\t"
			"psubw	%2, %%xmm4\n\t"
			"psubw	%3, %%xmm5\n\t"
			"psubw	%4, %%xmm6\n\t"
			"psubw	%5, %%xmm7\n"
		: :  "m" (EVAL_FEATURE[x].us[0]), "m" (EVAL_FEATURE[x].us[8]), "m" (EVAL_FEATURE[x].us[16]),
		"m" (EVAL_FEATURE[x].us[24]), "m" (EVAL_FEATURE[x].us[32]), "m" (EVAL_FEATURE[x].us[40]));
	}
	foreach_bit_32(x, fh) {
		__asm__ (
			"psubw	%0, %%xmm2\n\t"
			"psubw	%1, %%xmm3\n\t"
			"psubw	%2, %%xmm4\n\t"
			"psubw	%3, %%xmm5\n\t"
			"psubw	%4, %%xmm6\n\t"
			"psubw	%5, %%xmm7\n"
		: :  "m" (EVAL_FEATURE[x + 32].us[0]), "m" (EVAL_FEATURE[x + 32].us[8]), "m" (EVAL_FEATURE[x + 32].us[16]),
		"m" (EVAL_FEATURE[x + 32].us[24]), "m" (EVAL_FEATURE[x + 32].us[32]), "m" (EVAL_FEATURE[x + 32].us[40]));
	}

	__asm__ (
		"movdqu	%%xmm2, %0\n\t"
		"movdqu	%%xmm3, %1\n\t"
		"movdqu	%%xmm4, %2\n\t"
		"movdqu	%%xmm5, %3\n\t"
		"movdqu	%%xmm6, %4\n\t"
		"movdqu	%%xmm7, %5\n\t"
	: :  "m" (eval->feature.us[0]), "m" (eval->feature.us[8]), "m" (eval->feature.us[16]),
	"m" (eval->feature.us[24]), "m" (eval->feature.us[32]), "m" (eval->feature.us[40]));
}

static void eval_update_sse_1(Eval *eval, const Move *move)
{
	int	x = move->x;
	unsigned int	fl = (unsigned int) move->flipped;
	unsigned int	fh = (unsigned int) (move->flipped >> 32);

	__asm__ (
		"movdqu	%0, %%xmm2\n\t"		"movdqu	%1, %%xmm3\n\t"
		"psubw	%2, %%xmm2\n\t"		"psubw	%3, %%xmm3\n"
	: :  "m" (eval->feature.us[0]), "m" (eval->feature.us[8]), "m" (EVAL_FEATURE[x].us[0]),  "m" (EVAL_FEATURE[x].us[8]));
	__asm__ (
		"movdqu	%0, %%xmm4\n\t"		"movdqu	%1, %%xmm5\n\t"
		"psubw	%2, %%xmm4\n\t"		"psubw	%3, %%xmm5\n"
	: :  "m" (eval->feature.us[16]), "m" (eval->feature.us[24]), "m" (EVAL_FEATURE[x].us[16]),  "m" (EVAL_FEATURE[x].us[24]));
	__asm__ (
		"movdqu	%0, %%xmm6\n\t"		"movdqu	%1, %%xmm7\n\t"
		"psubw	%2, %%xmm6\n\t"		"psubw	%3, %%xmm7\n"
	: :  "m" (eval->feature.us[32]), "m" (eval->feature.us[40]), "m" (EVAL_FEATURE[x].us[32]),  "m" (EVAL_FEATURE[x].us[40]));

	foreach_bit_32(x, fl) {
		__asm__ (
			"paddw	%0, %%xmm2\n\t"
			"paddw	%1, %%xmm3\n\t"
			"paddw	%2, %%xmm4\n\t"
			"paddw	%3, %%xmm5\n\t"
			"paddw	%4, %%xmm6\n\t"
			"paddw	%5, %%xmm7\n"
		: :  "m" (EVAL_FEATURE[x].us[0]), "m" (EVAL_FEATURE[x].us[8]), "m" (EVAL_FEATURE[x].us[16]),
		"m" (EVAL_FEATURE[x].us[24]), "m" (EVAL_FEATURE[x].us[32]), "m" (EVAL_FEATURE[x].us[40]));
	}
	foreach_bit_32(x, fh) {
		__asm__ (
			"paddw	%0, %%xmm2\n\t"
			"paddw	%1, %%xmm3\n\t"
			"paddw	%2, %%xmm4\n\t"
			"paddw	%3, %%xmm5\n\t"
			"paddw	%4, %%xmm6\n\t"
			"paddw	%5, %%xmm7\n"
		: :  "m" (EVAL_FEATURE[x + 32].us[0]), "m" (EVAL_FEATURE[x + 32].us[8]), "m" (EVAL_FEATURE[x + 32].us[16]),
		"m" (EVAL_FEATURE[x + 32].us[24]), "m" (EVAL_FEATURE[x + 32].us[32]), "m" (EVAL_FEATURE[x + 32].us[40]));
	}

	__asm__ (
		"movdqu	%%xmm2, %0\n\t"
		"movdqu	%%xmm3, %1\n\t"
		"movdqu	%%xmm4, %2\n\t"
		"movdqu	%%xmm5, %3\n\t"
		"movdqu	%%xmm6, %4\n\t"
		"movdqu	%%xmm7, %5\n\t"
	: :  "m" (eval->feature.us[0]), "m" (eval->feature.us[8]), "m" (eval->feature.us[16]),
	"m" (eval->feature.us[24]), "m" (eval->feature.us[32]), "m" (eval->feature.us[40]));
}

static void eval_restore_sse_0(Eval *eval, const Move *move)
{
	int	x = move->x;
	unsigned int	fl = (unsigned int) move->flipped;
	unsigned int	fh = (unsigned int) (move->flipped >> 32);

	__asm__ (
		"movdqa	%2, %%xmm0\n\t"		"movdqa	%3, %%xmm1\n\t"
		"movdqu	%0, %%xmm2\n\t"		"movdqu	%1, %%xmm3\n\t"
		"psllw	$1, %%xmm0\n\t"		"psllw	$1, %%xmm1\n\t"
		"paddw	%%xmm0, %%xmm2\n\t"	"paddw	%%xmm1, %%xmm3\n"
	: :  "m" (eval->feature.us[0]), "m" (eval->feature.us[8]), "m" (EVAL_FEATURE[x].us[0]),  "m" (EVAL_FEATURE[x].us[8]));
	__asm__ (
		"movdqa	%2, %%xmm0\n\t"		"movdqa	%3, %%xmm1\n\t"
		"movdqu	%0, %%xmm4\n\t"		"movdqu	%1, %%xmm5\n\t"
		"psllw	$1, %%xmm0\n\t"		"psllw	$1, %%xmm1\n\t"
		"paddw	%%xmm0, %%xmm4\n\t"	"paddw	%%xmm1, %%xmm5\n"
	: :  "m" (eval->feature.us[16]), "m" (eval->feature.us[24]), "m" (EVAL_FEATURE[x].us[16]),  "m" (EVAL_FEATURE[x].us[24]));
	__asm__ (
		"movdqa	%2, %%xmm0\n\t"		"movdqa	%3, %%xmm1\n\t"
		"movdqu	%0, %%xmm6\n\t"		"movdqu	%1, %%xmm7\n\t"
		"psllw	$1, %%xmm0\n\t"		"psllw	$1, %%xmm1\n\t"
		"paddw	%%xmm0, %%xmm6\n\t"	"paddw	%%xmm1, %%xmm7\n"
	: :  "m" (eval->feature.us[32]), "m" (eval->feature.us[40]), "m" (EVAL_FEATURE[x].us[32]),  "m" (EVAL_FEATURE[x].us[40]));

	foreach_bit_32(x, fl) {
		__asm__ (
			"paddw	%0, %%xmm2\n\t"
			"paddw	%1, %%xmm3\n\t"
			"paddw	%2, %%xmm4\n\t"
			"paddw	%3, %%xmm5\n\t"
			"paddw	%4, %%xmm6\n\t"
			"paddw	%5, %%xmm7\n"
		: :  "m" (EVAL_FEATURE[x].us[0]), "m" (EVAL_FEATURE[x].us[8]), "m" (EVAL_FEATURE[x].us[16]),
		"m" (EVAL_FEATURE[x].us[24]), "m" (EVAL_FEATURE[x].us[32]), "m" (EVAL_FEATURE[x].us[40]));
	}
	foreach_bit_32(x, fh) {
		__asm__ (
			"paddw	%0, %%xmm2\n\t"
			"paddw	%1, %%xmm3\n\t"
			"paddw	%2, %%xmm4\n\t"
			"paddw	%3, %%xmm5\n\t"
			"paddw	%4, %%xmm6\n\t"
			"paddw	%5, %%xmm7\n"
		: :  "m" (EVAL_FEATURE[x + 32].us[0]), "m" (EVAL_FEATURE[x + 32].us[8]), "m" (EVAL_FEATURE[x + 32].us[16]),
		"m" (EVAL_FEATURE[x + 32].us[24]), "m" (EVAL_FEATURE[x + 32].us[32]), "m" (EVAL_FEATURE[x + 32].us[40]));
	}

	__asm__ (
		"movdqu	%%xmm2, %0\n\t"
		"movdqu	%%xmm3, %1\n\t"
		"movdqu	%%xmm4, %2\n\t"
		"movdqu	%%xmm5, %3\n\t"
		"movdqu	%%xmm6, %4\n\t"
		"movdqu	%%xmm7, %5\n\t"
	: :  "m" (eval->feature.us[0]), "m" (eval->feature.us[8]), "m" (eval->feature.us[16]),
	"m" (eval->feature.us[24]), "m" (eval->feature.us[32]), "m" (eval->feature.us[40]));
}

static void eval_restore_sse_1(Eval *eval, const Move *move)
{
	int	x = move->x;
	unsigned int	fl = (unsigned int) move->flipped;
	unsigned int	fh = (unsigned int) (move->flipped >> 32);

	__asm__ (
		"movdqu	%0, %%xmm2\n\t"		"movdqu	%1, %%xmm3\n\t"
		"paddw	%2, %%xmm2\n\t"		"paddw	%3, %%xmm3\n"
	: :  "m" (eval->feature.us[0]), "m" (eval->feature.us[8]), "m" (EVAL_FEATURE[x].us[0]),  "m" (EVAL_FEATURE[x].us[8]));
	__asm__ (
		"movdqu	%0, %%xmm4\n\t"		"movdqu	%1, %%xmm5\n\t"
		"paddw	%2, %%xmm4\n\t"		"paddw	%3, %%xmm5\n"
	: :  "m" (eval->feature.us[16]), "m" (eval->feature.us[24]), "m" (EVAL_FEATURE[x].us[16]),  "m" (EVAL_FEATURE[x].us[24]));
	__asm__ (
		"movdqu	%0, %%xmm6\n\t"		"movdqu	%1, %%xmm7\n\t"
		"paddw	%2, %%xmm6\n\t"		"paddw	%3, %%xmm7\n"
	: :  "m" (eval->feature.us[32]), "m" (eval->feature.us[40]), "m" (EVAL_FEATURE[x].us[32]),  "m" (EVAL_FEATURE[x].us[40]));

	foreach_bit_32(x, fl) {
		__asm__ (
			"psubw	%0, %%xmm2\n\t"
			"psubw	%1, %%xmm3\n\t"
			"psubw	%2, %%xmm4\n\t"
			"psubw	%3, %%xmm5\n\t"
			"psubw	%4, %%xmm6\n\t"
			"psubw	%5, %%xmm7\n"
		: :  "m" (EVAL_FEATURE[x].us[0]), "m" (EVAL_FEATURE[x].us[8]), "m" (EVAL_FEATURE[x].us[16]),
		"m" (EVAL_FEATURE[x].us[24]), "m" (EVAL_FEATURE[x].us[32]), "m" (EVAL_FEATURE[x].us[40]));
	}
	foreach_bit_32(x, fh) {
		__asm__ (
			"psubw	%0, %%xmm2\n\t"
			"psubw	%1, %%xmm3\n\t"
			"psubw	%2, %%xmm4\n\t"
			"psubw	%3, %%xmm5\n\t"
			"psubw	%4, %%xmm6\n\t"
			"psubw	%5, %%xmm7\n"
		: :  "m" (EVAL_FEATURE[x + 32].us[0]), "m" (EVAL_FEATURE[x + 32].us[8]), "m" (EVAL_FEATURE[x + 32].us[16]),
		"m" (EVAL_FEATURE[x + 32].us[24]), "m" (EVAL_FEATURE[x + 32].us[32]), "m" (EVAL_FEATURE[x + 32].us[40]));
	}

	__asm__ (
		"movdqu	%%xmm2, %0\n\t"
		"movdqu	%%xmm3, %1\n\t"
		"movdqu	%%xmm4, %2\n\t"
		"movdqu	%%xmm5, %3\n\t"
		"movdqu	%%xmm6, %4\n\t"
		"movdqu	%%xmm7, %5\n\t"
	: :  "m" (eval->feature.us[0]), "m" (eval->feature.us[8]), "m" (eval->feature.us[16]),
	"m" (eval->feature.us[24]), "m" (eval->feature.us[32]), "m" (eval->feature.us[40]));
}

#endif // hasSSE2

#ifdef hasSSE2

/**
 * @brief Set up evaluation features from a board.
 *
 * @param eval  Evaluation function.
 * @param board Board to setup features from.
 */
void eval_set(Eval *eval, const Board *board)
{
	int x;
	unsigned long long	b = board->player;
	static const EVAL_FEATURE_V EVAL_FEATURE_all_opponent = {{
		 9841,  9841,  9841,  9841, 29524, 29524, 29524, 29524, 29524, 29524, 29524, 29524, 29524, 29524, 29524, 29524,
		 3280,  3280,  3280,  3280,  3280,  3280,  3280,  3280,  3280,  3280,  3280,  3280,  3280,  3280,  1093,  1093,
		 1093,  1093,   364,   364,   364,   364,   121,   121,   121,   121,    40,    40,    40,    40,     0,     0
	}};
#ifdef __AVX2__
	__m256i	f0 = EVAL_FEATURE_all_opponent.v16[0];
	__m256i	f1 = EVAL_FEATURE_all_opponent.v16[1];
	__m256i	f2 = EVAL_FEATURE_all_opponent.v16[2];

	foreach_bit(x, b) {
		f0 = _mm256_sub_epi16(f0, EVAL_FEATURE[x].v16[0]);
		f1 = _mm256_sub_epi16(f1, EVAL_FEATURE[x].v16[1]);
		f2 = _mm256_sub_epi16(f2, EVAL_FEATURE[x].v16[2]);
	}
	b = ~(board->opponent | board->player);
	foreach_bit(x, b) {
		f0 = _mm256_add_epi16(f0, EVAL_FEATURE[x].v16[0]);
		f1 = _mm256_add_epi16(f1, EVAL_FEATURE[x].v16[1]);
		f2 = _mm256_add_epi16(f2, EVAL_FEATURE[x].v16[2]);
	}
	eval->feature.v16[0] = f0;
	eval->feature.v16[1] = f1;
	eval->feature.v16[2] = f2;

#else
	__m128i	f0 = EVAL_FEATURE_all_opponent.v8[0];
	__m128i	f1 = EVAL_FEATURE_all_opponent.v8[1];
	__m128i	f2 = EVAL_FEATURE_all_opponent.v8[2];
	__m128i	f3 = EVAL_FEATURE_all_opponent.v8[3];
	__m128i	f4 = EVAL_FEATURE_all_opponent.v8[4];
	__m128i	f5 = EVAL_FEATURE_all_opponent.v8[5];

	foreach_bit(x, b) {
		f0 = _mm_sub_epi16(f0, EVAL_FEATURE[x].v8[0]);
		f1 = _mm_sub_epi16(f1, EVAL_FEATURE[x].v8[1]);
		f2 = _mm_sub_epi16(f2, EVAL_FEATURE[x].v8[2]);
		f3 = _mm_sub_epi16(f3, EVAL_FEATURE[x].v8[3]);
		f4 = _mm_sub_epi16(f4, EVAL_FEATURE[x].v8[4]);
		f5 = _mm_sub_epi16(f5, EVAL_FEATURE[x].v8[5]);
	}
	b = ~(board->opponent | board->player);
	foreach_bit(x, b) {
		f0 = _mm_add_epi16(f0, EVAL_FEATURE[x].v8[0]);
		f1 = _mm_add_epi16(f1, EVAL_FEATURE[x].v8[1]);
		f2 = _mm_add_epi16(f2, EVAL_FEATURE[x].v8[2]);
		f3 = _mm_add_epi16(f3, EVAL_FEATURE[x].v8[3]);
		f4 = _mm_add_epi16(f4, EVAL_FEATURE[x].v8[4]);
		f5 = _mm_add_epi16(f5, EVAL_FEATURE[x].v8[5]);
	}

	eval->feature.v8[0] = f0;
	eval->feature.v8[1] = f1;
	eval->feature.v8[2] = f2;
	eval->feature.v8[3] = f3;
	eval->feature.v8[4] = f4;
	eval->feature.v8[5] = f5;
#endif
	eval->player = 0;
}

/**
 * @brief Swap player's feature.
 *
 * @param eval  Evaluation function.
 */
static void eval_swap(Eval *eval)
{
	eval->player ^= 1;
}

/**
 * @brief Update the features after a player's move.
 *
 * @param eval  Evaluation function.
 * @param move  Move.
 */
void eval_update(Eval *eval, const Move *move)
{
	assert(move->flipped);
	assert(WHITE == eval->player || BLACK == eval->player);
	if (eval->player)
		eval_update_sse_1(eval, move);
	else
		eval_update_sse_0(eval, move);
	eval_swap(eval);
}

/**
 * @brief Restore the features as before a player's move.
 *
 * @param eval  Evaluation function.
 * @param move  Move.
 */
void eval_restore(Eval *eval, const Move *move)
{
	assert(move->flipped);
	eval_swap(eval);
	assert(WHITE == eval->player || BLACK == eval->player);
	if (eval->player)
		eval_restore_sse_1(eval, move);
	else
		eval_restore_sse_0(eval, move);
}

/**
 * @brief Update/Restore the features after a passing move.
 *
 * @param eval  Evaluation function.
 */
void eval_pass(Eval *eval)
{
	eval_swap(eval);
}

#endif // hasSSE2
