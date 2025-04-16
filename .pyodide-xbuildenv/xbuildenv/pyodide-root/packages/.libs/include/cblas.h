#ifndef CBLAS_H
#define CBLAS_H

#include <stddef.h>
#include "openblas_config.h"

#ifdef __cplusplus
extern "C" {
	/* Assume C declarations for C++ */
#endif  /* __cplusplus */

/*Set the number of threads on runtime.*/
void openblas_set_num_threads(int num_threads);
void goto_set_num_threads(int num_threads);

/*Get the number of threads on runtime.*/
int openblas_get_num_threads(void);

/*Get the number of physical processors (cores).*/
int openblas_get_num_procs(void);

/*Get the build configure on runtime.*/
char* openblas_get_config(void);

/*Get the CPU corename on runtime.*/
char* openblas_get_corename(void);

#ifdef OPENBLAS_OS_LINUX
/* Sets thread affinity for OpenBLAS threads. `thread_idx` is in [0, openblas_get_num_threads()-1]. */
int openblas_setaffinity(int thread_idx, size_t cpusetsize, cpu_set_t* cpu_set);
/* Queries thread affinity for OpenBLAS threads. `thread_idx` is in [0, openblas_get_num_threads()-1]. */
int openblas_getaffinity(int thread_idx, size_t cpusetsize, cpu_set_t* cpu_set);
#endif

/* Get the parallelization type which is used by OpenBLAS */
int openblas_get_parallel(void);
/* OpenBLAS is compiled for sequential use  */
#define OPENBLAS_SEQUENTIAL  0
/* OpenBLAS is compiled using normal threading model */
#define OPENBLAS_THREAD  1
/* OpenBLAS is compiled using OpenMP threading model */
#define OPENBLAS_OPENMP 2


/*
 * Since all of GotoBlas was written without const,
 * we disable it at build time.
 */
#ifndef OPENBLAS_CONST
# define OPENBLAS_CONST const
#endif


#define CBLAS_INDEX size_t

typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
typedef CBLAS_ORDER CBLAS_LAYOUT;
	
float  cblas_sdsdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float *y, OPENBLAS_CONST blasint incy);
double cblas_dsdot (OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float *y, OPENBLAS_CONST blasint incy);
float  cblas_sdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float  *y, OPENBLAS_CONST blasint incy);
double cblas_ddot(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST double *y, OPENBLAS_CONST blasint incy);

openblas_complex_float  cblas_cdotu(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void  *y, OPENBLAS_CONST blasint incy);
openblas_complex_float  cblas_cdotc(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void  *y, OPENBLAS_CONST blasint incy);
openblas_complex_double cblas_zdotu(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void *y, OPENBLAS_CONST blasint incy);
openblas_complex_double cblas_zdotc(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void *y, OPENBLAS_CONST blasint incy);

int  cblas_cdotu_sub(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void  *y, OPENBLAS_CONST blasint incy, void  *ret);
int  cblas_cdotc_sub(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void  *y, OPENBLAS_CONST blasint incy, void  *ret);
int  cblas_zdotu_sub(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void *y, OPENBLAS_CONST blasint incy, void *ret);
int  cblas_zdotc_sub(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void *y, OPENBLAS_CONST blasint incy, void *ret);

float  cblas_sasum (OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
double cblas_dasum (OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
float  cblas_scasum(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
double cblas_dzasum(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

float  cblas_ssum (OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
double cblas_dsum (OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
float  cblas_scsum(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
double cblas_dzsum(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

float  cblas_snrm2 (OPENBLAS_CONST blasint N, OPENBLAS_CONST float  *X, OPENBLAS_CONST blasint incX);
double cblas_dnrm2 (OPENBLAS_CONST blasint N, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX);
float  cblas_scnrm2(OPENBLAS_CONST blasint N, OPENBLAS_CONST void  *X, OPENBLAS_CONST blasint incX);
double cblas_dznrm2(OPENBLAS_CONST blasint N, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX);

CBLAS_INDEX cblas_isamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_idamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_icamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_izamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

CBLAS_INDEX cblas_isamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_idamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_icamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_izamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

CBLAS_INDEX cblas_ismax(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_idmax(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_icmax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_izmax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

CBLAS_INDEX cblas_ismin(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_idmin(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_icmin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
CBLAS_INDEX cblas_izmin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

int cblas_saxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
int cblas_daxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
int cblas_caxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);
int cblas_zaxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);

int cblas_scopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
int cblas_dcopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
int cblas_ccopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);
int cblas_zcopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);

int cblas_sswap(OPENBLAS_CONST blasint n, float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
int cblas_dswap(OPENBLAS_CONST blasint n, double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
int cblas_cswap(OPENBLAS_CONST blasint n, void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);
int cblas_zswap(OPENBLAS_CONST blasint n, void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);

int cblas_srot(OPENBLAS_CONST blasint N, float *X, OPENBLAS_CONST blasint incX, float *Y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST float c, OPENBLAS_CONST float s);
int cblas_drot(OPENBLAS_CONST blasint N, double *X, OPENBLAS_CONST blasint incX, double *Y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST double c, OPENBLAS_CONST double  s);
int cblas_csrot(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST float c, OPENBLAS_CONST float s);
int cblas_zdrot(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST double c, OPENBLAS_CONST double s);

int cblas_srotg(float *a, float *b, float *c, float *s);
int cblas_drotg(double *a, double *b, double *c, double *s);
int cblas_crotg(void *a, void *b, float *c, void *s);
int cblas_zrotg(void *a, void *b, double *c, void *s);


int cblas_srotm(OPENBLAS_CONST blasint N, float *X, OPENBLAS_CONST blasint incX, float *Y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST float *P);
int cblas_drotm(OPENBLAS_CONST blasint N, double *X, OPENBLAS_CONST blasint incX, double *Y, OPENBLAS_CONST blasint incY, OPENBLAS_CONST double *P);

int cblas_srotmg(float *d1, float *d2, float *b1, OPENBLAS_CONST float b2, float *P);
int cblas_drotmg(double *d1, double *d2, double *b1, OPENBLAS_CONST double b2, double *P);

int cblas_sscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, float *X, OPENBLAS_CONST blasint incX);
int cblas_dscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, double *X, OPENBLAS_CONST blasint incX);
int cblas_cscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, void *X, OPENBLAS_CONST blasint incX);
int cblas_zscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, void *X, OPENBLAS_CONST blasint incX);
int cblas_csscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, void *X, OPENBLAS_CONST blasint incX);
int cblas_zdscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, void *X, OPENBLAS_CONST blasint incX);

int cblas_sgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST float beta,  float  *y, OPENBLAS_CONST blasint incy);
int cblas_dgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST double alpha, OPENBLAS_CONST double  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST double  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST double beta,  double  *y, OPENBLAS_CONST blasint incy);
int cblas_cgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST void *beta,  void  *y, OPENBLAS_CONST blasint incy);
int cblas_zgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST void *beta,  void  *y, OPENBLAS_CONST blasint incy);

int cblas_sger (OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST float   alpha, OPENBLAS_CONST float  *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float  *Y, OPENBLAS_CONST blasint incY, float  *A, OPENBLAS_CONST blasint lda);
int cblas_dger (OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST double  alpha, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double *Y, OPENBLAS_CONST blasint incY, double *A, OPENBLAS_CONST blasint lda);
int cblas_cgeru(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void  *alpha, OPENBLAS_CONST void  *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void  *Y, OPENBLAS_CONST blasint incY, void  *A, OPENBLAS_CONST blasint lda);
int cblas_cgerc(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void  *alpha, OPENBLAS_CONST void  *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void  *Y, OPENBLAS_CONST blasint incY, void  *A, OPENBLAS_CONST blasint lda);
int cblas_zgeru(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *A, OPENBLAS_CONST blasint lda);
int cblas_zgerc(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *A, OPENBLAS_CONST blasint lda);

int cblas_strsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *X, OPENBLAS_CONST blasint incX);
int cblas_dtrsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *X, OPENBLAS_CONST blasint incX);
int cblas_ctrsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);
int cblas_ztrsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);

int cblas_strmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *X, OPENBLAS_CONST blasint incX);
int cblas_dtrmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *X, OPENBLAS_CONST blasint incX);
int cblas_ctrmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);
int cblas_ztrmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);

int cblas_ssyr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, float *A, OPENBLAS_CONST blasint lda);
int cblas_dsyr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, double *A, OPENBLAS_CONST blasint lda);
int cblas_cher(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, void *A, OPENBLAS_CONST blasint lda);
int cblas_zher(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, void *A, OPENBLAS_CONST blasint lda);

int cblas_ssyr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo,OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *X,
                OPENBLAS_CONST blasint incX, OPENBLAS_CONST float *Y, OPENBLAS_CONST blasint incY, float *A, OPENBLAS_CONST blasint lda);
int cblas_dsyr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *X,
                OPENBLAS_CONST blasint incX, OPENBLAS_CONST double *Y, OPENBLAS_CONST blasint incY, double *A, OPENBLAS_CONST blasint lda);
int cblas_cher2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX,
                OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *A, OPENBLAS_CONST blasint lda);
int cblas_zher2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX,
                OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *A, OPENBLAS_CONST blasint lda);

int cblas_sgbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST blasint KL, OPENBLAS_CONST blasint KU, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float beta, float *Y, OPENBLAS_CONST blasint incY);
int cblas_dgbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST blasint KL, OPENBLAS_CONST blasint KU, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double beta, double *Y, OPENBLAS_CONST blasint incY);
int cblas_cgbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST blasint KL, OPENBLAS_CONST blasint KU, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);
int cblas_zgbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST blasint KL, OPENBLAS_CONST blasint KU, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);

int cblas_ssbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float beta, float *Y, OPENBLAS_CONST blasint incY);
int cblas_dsbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double beta, double *Y, OPENBLAS_CONST blasint incY);


int cblas_stbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *X, OPENBLAS_CONST blasint incX);
int cblas_dtbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *X, OPENBLAS_CONST blasint incX);
int cblas_ctbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);
int cblas_ztbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);

int cblas_stbsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *X, OPENBLAS_CONST blasint incX);
int cblas_dtbsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *X, OPENBLAS_CONST blasint incX);
int cblas_ctbsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);
int cblas_ztbsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *X, OPENBLAS_CONST blasint incX);

int cblas_stpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST float *Ap, float *X, OPENBLAS_CONST blasint incX);
int cblas_dtpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST double *Ap, double *X, OPENBLAS_CONST blasint incX);
int cblas_ctpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST void *Ap, void *X, OPENBLAS_CONST blasint incX);
int cblas_ztpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST void *Ap, void *X, OPENBLAS_CONST blasint incX);

int cblas_stpsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST float *Ap, float *X, OPENBLAS_CONST blasint incX);
int cblas_dtpsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST double *Ap, double *X, OPENBLAS_CONST blasint incX);
int cblas_ctpsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST void *Ap, void *X, OPENBLAS_CONST blasint incX);
int cblas_ztpsv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_DIAG Diag,
                 OPENBLAS_CONST blasint N, OPENBLAS_CONST void *Ap, void *X, OPENBLAS_CONST blasint incX);

int cblas_ssymv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float beta, float *Y, OPENBLAS_CONST blasint incY);
int cblas_dsymv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double beta, double *Y, OPENBLAS_CONST blasint incY);
int cblas_chemv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);
int cblas_zhemv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A,
                 OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);


int cblas_sspmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *Ap,
                 OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float beta, float *Y, OPENBLAS_CONST blasint incY);
int cblas_dspmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *Ap,
                 OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double beta, double *Y, OPENBLAS_CONST blasint incY);

int cblas_sspr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, float *Ap);
int cblas_dspr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, double *Ap);

int cblas_chpr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, void *A);
int cblas_zhpr(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST void *X,OPENBLAS_CONST blasint incX, void *A);

int cblas_sspr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST float *Y, OPENBLAS_CONST blasint incY, float *A);
int cblas_dspr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST double *Y, OPENBLAS_CONST blasint incY, double *A);
int cblas_chpr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *Ap);
int cblas_zhpr2(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *Y, OPENBLAS_CONST blasint incY, void *Ap);

int cblas_chbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);
int cblas_zhbmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);

int cblas_chpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *Ap, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);
int cblas_zhpmv(OPENBLAS_CONST enum CBLAS_ORDER order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint N,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *Ap, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX, OPENBLAS_CONST void *beta, void *Y, OPENBLAS_CONST blasint incY);

int cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
int cblas_dgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
int cblas_cgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
int cblas_cgemm3m(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
int cblas_zgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
int cblas_zgemm3m(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);


int cblas_ssymm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
int cblas_dsymm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
int cblas_csymm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
int cblas_zsymm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);

int cblas_ssyrk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
int cblas_dsyrk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
int cblas_csyrk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
int cblas_zsyrk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		 OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);

int cblas_ssyr2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		  OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
int cblas_dsyr2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		  OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
int cblas_csyr2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		  OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
int cblas_zsyr2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans,
		  OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);

int cblas_strmm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *B, OPENBLAS_CONST blasint ldb);
int cblas_dtrmm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *B, OPENBLAS_CONST blasint ldb);
int cblas_ctrmm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *B, OPENBLAS_CONST blasint ldb);
int cblas_ztrmm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *B, OPENBLAS_CONST blasint ldb);

int cblas_strsm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, float *B, OPENBLAS_CONST blasint ldb);
int cblas_dtrsm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, double *B, OPENBLAS_CONST blasint ldb);
int cblas_ctrsm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *B, OPENBLAS_CONST blasint ldb);
int cblas_ztrsm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
                 OPENBLAS_CONST enum CBLAS_DIAG Diag, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, void *B, OPENBLAS_CONST blasint ldb);

int cblas_chemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);
int cblas_zhemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_SIDE Side, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N,
                 OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST void *beta, void *C, OPENBLAS_CONST blasint ldc);

int cblas_cherk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                 OPENBLAS_CONST float alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float beta, void *C, OPENBLAS_CONST blasint ldc);
int cblas_zherk(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                 OPENBLAS_CONST double alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double beta, void *C, OPENBLAS_CONST blasint ldc);

int cblas_cher2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                  OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, void *C, OPENBLAS_CONST blasint ldc);
int cblas_zher2k(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_UPLO Uplo, OPENBLAS_CONST enum CBLAS_TRANSPOSE Trans, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
                  OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, void *C, OPENBLAS_CONST blasint ldc);

int cblas_xerbla(blasint p, char *rout, char *form, ...);

/*** BLAS extensions ***/

int cblas_saxpby(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx,OPENBLAS_CONST float beta, float *y, OPENBLAS_CONST blasint incy);

int cblas_daxpby(OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx,OPENBLAS_CONST double beta, double *y, OPENBLAS_CONST blasint incy);

int cblas_caxpby(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx,OPENBLAS_CONST void *beta, void *y, OPENBLAS_CONST blasint incy);

int cblas_zaxpby(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *alpha, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx,OPENBLAS_CONST void *beta, void *y, OPENBLAS_CONST blasint incy);

int cblas_somatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float calpha, OPENBLAS_CONST float *a, 
		     OPENBLAS_CONST blasint clda, float *b, OPENBLAS_CONST blasint cldb); 
int cblas_domatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double calpha, OPENBLAS_CONST double *a,
		     OPENBLAS_CONST blasint clda, double *b, OPENBLAS_CONST blasint cldb); 
int cblas_comatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float* calpha, OPENBLAS_CONST float* a, 
		     OPENBLAS_CONST blasint clda, float*b, OPENBLAS_CONST blasint cldb); 
int cblas_zomatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double* calpha, OPENBLAS_CONST double* a, 
		     OPENBLAS_CONST blasint clda,  double *b, OPENBLAS_CONST blasint cldb); 

int cblas_simatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float calpha, float *a, 
		     OPENBLAS_CONST blasint clda, OPENBLAS_CONST blasint cldb); 
int cblas_dimatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double calpha, double *a,
		     OPENBLAS_CONST blasint clda, OPENBLAS_CONST blasint cldb); 
int cblas_cimatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float* calpha, float* a, 
		     OPENBLAS_CONST blasint clda, OPENBLAS_CONST blasint cldb); 
int cblas_zimatcopy(OPENBLAS_CONST enum CBLAS_ORDER CORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE CTRANS, OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double* calpha, double* a, 
		     OPENBLAS_CONST blasint clda, OPENBLAS_CONST blasint cldb); 

int cblas_sgeadd(OPENBLAS_CONST enum CBLAS_ORDER CORDER,OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float calpha, float *a, OPENBLAS_CONST blasint clda, OPENBLAS_CONST float cbeta, 
		  float *c, OPENBLAS_CONST blasint cldc); 
int cblas_dgeadd(OPENBLAS_CONST enum CBLAS_ORDER CORDER,OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double calpha, double *a, OPENBLAS_CONST blasint clda, OPENBLAS_CONST double cbeta, 
		  double *c, OPENBLAS_CONST blasint cldc); 
int cblas_cgeadd(OPENBLAS_CONST enum CBLAS_ORDER CORDER,OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST float *calpha, float *a, OPENBLAS_CONST blasint clda, OPENBLAS_CONST float *cbeta, 
		  float *c, OPENBLAS_CONST blasint cldc); 
int cblas_zgeadd(OPENBLAS_CONST enum CBLAS_ORDER CORDER,OPENBLAS_CONST blasint crows, OPENBLAS_CONST blasint ccols, OPENBLAS_CONST double *calpha, double *a, OPENBLAS_CONST blasint clda, OPENBLAS_CONST double *cbeta, 
		  double *c, OPENBLAS_CONST blasint cldc); 

/*** BFLOAT16 and INT8 extensions ***/
/* convert float array to BFLOAT16 array by rounding */
int   cblas_sbstobf16(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *in, OPENBLAS_CONST blasint incin, bfloat16 *out, OPENBLAS_CONST blasint incout);
/* convert double array to BFLOAT16 array by rounding */
int   cblas_sbdtobf16(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *in, OPENBLAS_CONST blasint incin, bfloat16 *out, OPENBLAS_CONST blasint incout);
/* convert BFLOAT16 array to float array */
int   cblas_sbf16tos(OPENBLAS_CONST blasint n, OPENBLAS_CONST bfloat16 *in, OPENBLAS_CONST blasint incin, float  *out, OPENBLAS_CONST blasint incout);
/* convert BFLOAT16 array to double array */
int   cblas_dbf16tod(OPENBLAS_CONST blasint n, OPENBLAS_CONST bfloat16 *in, OPENBLAS_CONST blasint incin, double *out, OPENBLAS_CONST blasint incout);
/* dot production of BFLOAT16 input arrays, and output as float */
float  cblas_sbdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST bfloat16 *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST bfloat16 *y, OPENBLAS_CONST blasint incy);
int   cblas_sbgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 *a, OPENBLAS_CONST blasint lda, OPENBLAS_CONST bfloat16 *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float beta, float *y, OPENBLAS_CONST blasint incy);

int   cblas_sbgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		    OPENBLAS_CONST float alpha, OPENBLAS_CONST bfloat16 *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST bfloat16 *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
#ifdef __cplusplus
}
#endif  /* __cplusplus */

#endif
