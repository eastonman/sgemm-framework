#include <include/sgemm.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <x86intrin.h> // _AVX512

using namespace std;

// Currently use Row Major storage

#define _A(i, j) a[(i)*lda + (j)]
#define _B(i, j) b[(i)*ldb + (j)]
#define _C(i, j) c[(i)*ldc + (j)]

/* Block sizes */
#define _A_BLOCK_SIZE 1024
#define _B_BLOCK_SIZE 128

namespace lib
{

    static void _AddDot4x64(int, const float *, int, const float *, int, float *, int);
    static void InnerKernel(int m, int n, int k, float *a, int lda,
                            float *b, int ldb,
                            float *c, int ldc);
    static void PackMatrixB(int, float *, int, float *);
    static float *packedB = nullptr;

    void sgemm(int m, int n, int k, float *a, int lda,
               float *b, int ldb,
               float *c, int ldc)
    {
        // size_t i, j;
        // for (i = 0; i < m; i += 4)
        // {
        //     for (j = 0; j < n; j += 64)
        //     {
        //         _AddDot4x64(k, &_A(i, 0), lda, &_B(0, j), ldb, &_C(i, j), ldc);
        //     }
        // }
        packedB = (float *)_mm_malloc(n * k * sizeof(float), 64);
        // InnerKernel(m, n, k, a, lda, b, ldb, c, ldc);

        int i, j, p, pb, ib;

        for (p = 0; p < k; p += _A_BLOCK_SIZE)
        {
            pb = min(k - p, _A_BLOCK_SIZE);
            for (i = 0; i < n; i += _B_BLOCK_SIZE)
            {
                ib = min(n - i, _B_BLOCK_SIZE);
                InnerKernel(m, ib, pb, &_A(0, p), lda, &_B(p, i), ldb, &_C(0, i), ldc);
            }
        }
    }

    static void InnerKernel(int m, int n, int k, float *a, int lda,
                            float *b, int ldb,
                            float *c, int ldc)
    {
        int i, j;

        for (i = 0; i < m; i += 4)
        {
            for (j = 0; j < n; j += 64)
            {
                if (i == 0)
                {
                    PackMatrixB(k, &_B(0, j), ldb, &packedB[j * k]);
                }
                _AddDot4x64(k, &_A(i, 0), lda, &packedB[j * k], 64, &_C(i, j), ldc);
            }
        }
    }

    static inline void PackMatrixB(int k, float *b, int ldb, float *to)
    {
        for (size_t j = 0; j < k; j++)
        {
            std::memcpy(to + 64 * j, &_B(j, 0), 64 * sizeof(float));
        }
    }

    static inline void _AddDot4x64(int k, const float *a, int lda, const float *b, int ldb, float *c, int ldc)
    {
        __m512 c_reg[16];
        // Initialize C registers
#pragma GCC unroll 16
        for (size_t i = 0; i < 16; i++)
        {
            c_reg[i] = _mm512_setzero_ps();
        }

        __m512 a_reg[4];
        __m512 b_reg[4];

#pragma GCC unroll 4
        for (size_t sliding_idx = 0; sliding_idx < k - 1; sliding_idx++)
        {
            // Load B reg
#pragma GCC unroll 4
            for (size_t i = 0; i < 4; i++)
            {
                b_reg[i] = _mm512_load_ps(&_B(sliding_idx, i * 16));
            }

            // 4 A registers
#pragma GCC unroll 4
            for (size_t i = 0; i < 4; i++)
            {
                // Broadcast 1 to 16
                a_reg[i] = _mm512_set1_ps(_A(i, sliding_idx));
            }

            // Do multiply
#pragma GCC unroll 4
            for (size_t i = 0; i < 4; i++)
            {
                c_reg[i * 4 + 0] += a_reg[i] * b_reg[0];
                c_reg[i * 4 + 1] += a_reg[i] * b_reg[1];
                c_reg[i * 4 + 2] += a_reg[i] * b_reg[2];
                c_reg[i * 4 + 3] += a_reg[i] * b_reg[3];
            }
        }

        // Load B reg
#pragma GCC unroll 4
        for (size_t i = 0; i < 4; i++)
        {
            b_reg[i] = _mm512_load_ps(&_B(k - 1, i * 16));
        }

        // 4 A registers
#pragma GCC unroll 4
        for (size_t i = 0; i < 4; i++)
        {
            // Broadcast 1 to 16
            a_reg[i] = _mm512_set1_ps(_A(i, k - 1));
        }

        // Do multiply
        __m512 c_tmp[4];
#pragma GCC unroll 4
        for (size_t i = 0; i < 4; i++)
        {
#pragma GCC unroll 4
            for (size_t j = 0; j < 4; j++)
            {
                c_tmp[j] = _mm512_load_ps(&_C(i, j * 16));
                c_reg[i * 4 + j] += a_reg[i] * b_reg[j];
                c_tmp[j] += c_reg[i * 4 + j];
                _mm512_store_ps(&_C(i, j * 16), c_tmp[j]);
            }
            // c_reg[i * 4 + 1] += a_reg[i] * b_reg[1];
            // c_reg[i * 4 + 2] += a_reg[i] * b_reg[2];
            // c_reg[i * 4 + 3] += a_reg[i] * b_reg[3];
        }

        // Store c_reg back to memory
        //         for (size_t i = 0; i < 4; i++)
        //         {
        // #pragma GCC unroll 4
        //             for (size_t j = 0; j < 4; j++)
        //             {
        //                 __m512 c_tmp = _mm512_load_ps(&_C(i, j * 16));
        //                 c_tmp += c_reg[i * 4 + j];
        //                 _mm512_store_ps(&_C(i, j * 16), c_tmp);
        //             }
        //         }
    }
}