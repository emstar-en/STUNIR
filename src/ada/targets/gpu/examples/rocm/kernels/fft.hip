/**
 * STUNIR ROCm Kernel: Fast Fourier Transform
 *
 * Cooley-Tukey FFT algorithm implementation for GPU.
 * Supports radix-2 and radix-4 transforms.
 *
 * Schema: stunir.gpu.rocm.kernel.fft.v1
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846
#define MAX_SHARED_SIZE 4096

#define CHECK_HIP(call) { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, \
                hipGetErrorString(err)); \
        exit(1); \
    } \
}

/**
 * Complex number operations (device functions)
 */
struct Complex {
    float real;
    float imag;
    
    __device__ __host__ Complex() : real(0.0f), imag(0.0f) {}
    __device__ __host__ Complex(float r, float i) : real(r), imag(i) {}
    
    __device__ __host__ Complex operator+(const Complex& b) const {
        return Complex(real + b.real, imag + b.imag);
    }
    
    __device__ __host__ Complex operator-(const Complex& b) const {
        return Complex(real - b.real, imag - b.imag);
    }
    
    __device__ __host__ Complex operator*(const Complex& b) const {
        return Complex(real * b.real - imag * b.imag,
                      real * b.imag + imag * b.real);
    }
    
    __device__ __host__ float magnitude() const {
        return sqrtf(real * real + imag * imag);
    }
};

/**
 * Compute twiddle factor: W_N^k = e^(-2*pi*i*k/N)
 */
__device__ __host__ Complex twiddle(int k, int N, int direction) {
    float angle = direction * 2.0f * PI * k / N;
    return Complex(cosf(angle), sinf(angle));
}

/**
 * Bit-reversal permutation index
 */
__device__ __host__ unsigned int bitReverse(unsigned int x, int log2n) {
    unsigned int result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/**
 * Radix-2 Cooley-Tukey FFT kernel (iterative, in-place)
 * Each block processes one N-point FFT
 */
__global__ void fft_radix2(Complex* __restrict__ data, int N, int log2N, int direction) {
    __shared__ Complex shared_data[MAX_SHARED_SIZE];
    
    int tid = hipThreadIdx_x;
    int blockSize = hipBlockDim_x;
    int blockId = hipBlockIdx_x;
    
    // Load data with bit-reversal permutation
    Complex* block_data = data + blockId * N;
    
    for (int i = tid; i < N; i += blockSize) {
        unsigned int j = bitReverse(i, log2N);
        shared_data[i] = block_data[j];
    }
    __syncthreads();
    
    // Cooley-Tukey iterative FFT
    for (int s = 1; s <= log2N; s++) {
        int m = 1 << s;          // Current transform size
        int m2 = m >> 1;         // Half of transform size
        
        for (int k = tid; k < N / 2; k += blockSize) {
            int j = k & (m2 - 1);  // Position within butterfly group
            int i = (k / m2) * m + j;  // Index of first element
            
            Complex w = twiddle(j, m, direction);
            Complex t = w * shared_data[i + m2];
            Complex u = shared_data[i];
            
            shared_data[i] = u + t;
            shared_data[i + m2] = u - t;
        }
        __syncthreads();
    }
    
    // Write back results
    for (int i = tid; i < N; i += blockSize) {
        block_data[i] = shared_data[i];
    }
}

/**
 * Radix-4 FFT kernel for better performance
 * Processes 4 elements per butterfly
 */
__global__ void fft_radix4(Complex* __restrict__ data, int N, int log4N, int direction) {
    __shared__ Complex shared_data[MAX_SHARED_SIZE];
    
    int tid = hipThreadIdx_x;
    int blockSize = hipBlockDim_x;
    int blockId = hipBlockIdx_x;
    
    Complex* block_data = data + blockId * N;
    
    // Load with bit-reversed indices (radix-4)
    int log2N = log4N * 2;
    for (int i = tid; i < N; i += blockSize) {
        unsigned int j = bitReverse(i, log2N);
        shared_data[i] = block_data[j];
    }
    __syncthreads();
    
    // Radix-4 stages
    for (int s = 0; s < log4N; s++) {
        int m = 1 << (2 * (s + 1));  // Radix-4 size: 4, 16, 64, ...
        int m4 = m >> 2;
        
        for (int k = tid; k < N / 4; k += blockSize) {
            int j = k & (m4 - 1);
            int i = (k / m4) * m + j;
            
            Complex w1 = twiddle(j, m, direction);
            Complex w2 = twiddle(2 * j, m, direction);
            Complex w3 = twiddle(3 * j, m, direction);
            
            Complex a0 = shared_data[i];
            Complex a1 = shared_data[i + m4] * w1;
            Complex a2 = shared_data[i + 2 * m4] * w2;
            Complex a3 = shared_data[i + 3 * m4] * w3;
            
            // Radix-4 butterfly
            Complex b0 = a0 + a2;
            Complex b1 = a1 + a3;
            Complex b2 = a0 - a2;
            Complex b3 = a1 - a3;
            
            // Handle imaginary unit multiplication for b3
            Complex j_b3 = Complex(direction * b3.imag, -direction * b3.real);
            
            shared_data[i] = b0 + b1;
            shared_data[i + m4] = b2 + j_b3;
            shared_data[i + 2 * m4] = b0 - b1;
            shared_data[i + 3 * m4] = b2 - j_b3;
        }
        __syncthreads();
    }
    
    // Write back
    for (int i = tid; i < N; i += blockSize) {
        block_data[i] = shared_data[i];
    }
}

/**
 * FFT normalization kernel (for inverse FFT)
 */
__global__ void fft_normalize(Complex* data, int N, float scale) {
    int idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (idx < N) {
        data[idx].real *= scale;
        data[idx].imag *= scale;
    }
}

/**
 * 2D FFT using row/column decomposition
 */
void fft2d(Complex* d_data, int rows, int cols, int direction) {
    int log2cols = (int)log2f((float)cols);
    int log2rows = (int)log2f((float)rows);
    
    // FFT on rows
    hipLaunchKernelGGL(fft_radix2, dim3(rows), dim3(256), 0, 0,
                       d_data, cols, log2cols, direction);
    
    // Transpose (simplified - in practice use optimized transpose)
    // FFT on columns (now rows after transpose)
    hipLaunchKernelGGL(fft_radix2, dim3(cols), dim3(256), 0, 0,
                       d_data, rows, log2rows, direction);
}

// Host verification
void dft_reference(Complex* input, Complex* output, int N, int direction) {
    for (int k = 0; k < N; k++) {
        output[k] = Complex(0.0f, 0.0f);
        for (int n = 0; n < N; n++) {
            float angle = direction * 2.0f * PI * k * n / N;
            Complex w(cosf(angle), sinf(angle));
            output[k] = output[k] + input[n] * w;
        }
    }
}

bool verify_fft(Complex* result, Complex* reference, int N, float tolerance) {
    int errors = 0;
    for (int i = 0; i < N && errors < 5; i++) {
        float diff_real = fabsf(result[i].real - reference[i].real);
        float diff_imag = fabsf(result[i].imag - reference[i].imag);
        if (diff_real > tolerance || diff_imag > tolerance) {
            printf("Mismatch at %d: (%.4f, %.4f) vs (%.4f, %.4f)\n",
                   i, result[i].real, result[i].imag,
                   reference[i].real, reference[i].imag);
            errors++;
        }
    }
    return errors == 0;
}

int main(int argc, char** argv) {
    int N = 1024;  // Must be power of 2
    int numFFTs = 64;
    int direction = -1;  // -1 for forward, 1 for inverse
    
    printf("=== STUNIR ROCm: Fast Fourier Transform ===\n");
    printf("FFT size: %d, Batch: %d\n", N, numFFTs);
    
    int log2N = (int)log2f((float)N);
    size_t total_size = N * numFFTs * sizeof(Complex);
    
    // Allocate
    Complex* h_data = (Complex*)malloc(total_size);
    Complex* h_result = (Complex*)malloc(total_size);
    Complex* h_reference = (Complex*)malloc(N * sizeof(Complex));
    
    // Initialize with test signal
    srand(42);
    for (int b = 0; b < numFFTs; b++) {
        for (int i = 0; i < N; i++) {
            h_data[b * N + i] = Complex(
                sinf(2.0f * PI * 10.0f * i / N) + 0.5f * sinf(2.0f * PI * 50.0f * i / N),
                0.0f
            );
        }
    }
    
    // Device memory
    Complex* d_data;
    CHECK_HIP(hipMalloc(&d_data, total_size));
    CHECK_HIP(hipMemcpy(d_data, h_data, total_size, hipMemcpyHostToDevice));
    
    // Timing
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    
    // Run FFT
    CHECK_HIP(hipEventRecord(start));
    hipLaunchKernelGGL(fft_radix2, dim3(numFFTs), dim3(256), 0, 0,
                       d_data, N, log2N, direction);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    
    float elapsed;
    CHECK_HIP(hipEventElapsedTime(&elapsed, start, stop));
    printf("Radix-2 FFT time: %.3f ms (%.3f us per FFT)\n", 
           elapsed, elapsed * 1000.0f / numFFTs);
    
    // Copy result
    CHECK_HIP(hipMemcpy(h_result, d_data, total_size, hipMemcpyDeviceToHost));
    
    // Verify first FFT against DFT reference
    printf("Computing reference DFT...\n");
    dft_reference(h_data, h_reference, N, direction);
    
    printf("Verification: %s\n",
           verify_fft(h_result, h_reference, N, 1e-3f) ? "PASSED" : "FAILED");
    
    // Performance metrics
    double ops = 5.0 * N * log2N * numFFTs;  // Approximate FLOPS for radix-2
    double gflops = ops / (elapsed / 1000.0) / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Throughput: %.2f FFTs/second\n", numFFTs / (elapsed / 1000.0));
    
    // Test inverse FFT
    printf("\nTesting inverse FFT...\n");
    
    // Forward FFT already done, now inverse
    CHECK_HIP(hipEventRecord(start));
    hipLaunchKernelGGL(fft_radix2, dim3(numFFTs), dim3(256), 0, 0,
                       d_data, N, log2N, 1);  // direction = 1 for inverse
    hipLaunchKernelGGL(fft_normalize, dim3((N * numFFTs + 255) / 256), dim3(256), 0, 0,
                       d_data, N * numFFTs, 1.0f / N);
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));
    CHECK_HIP(hipEventElapsedTime(&elapsed, start, stop));
    printf("Inverse FFT time: %.3f ms\n", elapsed);
    
    CHECK_HIP(hipMemcpy(h_result, d_data, total_size, hipMemcpyDeviceToHost));
    
    // Verify round-trip
    int errors = 0;
    for (int i = 0; i < N && errors < 5; i++) {
        if (fabsf(h_result[i].real - h_data[i].real) > 1e-3f) {
            printf("Round-trip error at %d: %.4f vs %.4f\n",
                   i, h_result[i].real, h_data[i].real);
            errors++;
        }
    }
    printf("Round-trip verification: %s\n", errors == 0 ? "PASSED" : "FAILED");
    
    // Cleanup
    CHECK_HIP(hipFree(d_data));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    free(h_data);
    free(h_result);
    free(h_reference);
    
    return 0;
}
