#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define yoko 2048
#define tate 2048
//#define pie 3.14159265358979323846

#pragma pack(push,1)
typedef struct tagBITMAPFILEHEADER {
    unsigned short bfType;
    unsigned int bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int bf0ffBits;
} BITMAPFILEHEADER;
#pragma pack(pop)

typedef struct tagBITMAPINFOHEADER {
    unsigned int biSize;
    int biWidth;
    int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned int biCirUsed;
    unsigned int biCirImportant;
} BITMAPINFOHEADER;

typedef struct tagRGBQUAD {
    unsigned char rgbBlue;
    unsigned char rgbGreen;
    unsigned char rgbRed;
    unsigned char rgbReserved;
} RGBQUAD;

unsigned char img[yoko][tate];
double kekka[yoko][tate] = {0.0};

// CUDAカーネル関数
__global__ void keisan(float *x, float *y, double *z, double *kekka, int n, double K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < yoko && j < tate) {
        double dx, dy;
        for (int m = 0; m < n; m++) {
            dx = (double)(x[m] - i);
            dy = (double)(y[m] - j);
            kekka[i * tate + j] += cos(K * sqrt(dx * dx + dy * dy + z[m] * z[m]));
        }
    }
}

int main() {
    FILE *fpten;

    int n, i, j;
    float *h_x, *h_y;
    double *h_z;
    double min_kekka = 0.0, max_kekka = 0.0, mid_kekka = 0.0;

    // ファイル読み込み
    fpten = fopen("chess_000.3df", "rb");
    fread(&n, sizeof(int), 1, fpten);

    // ホストメモリ割り当て
    h_x = (float*)malloc(n * sizeof(float));
    h_y = (float*)malloc(n * sizeof(float));
    h_z = (double*)malloc(n * sizeof(double));

    for (i = 0; i < n; i++) {
        float tx, ty, tz;
        fread(&tx, sizeof(float), 1, fpten);
        fread(&ty, sizeof(float), 1, fpten);
        fread(&tz, sizeof(float), 1, fpten);
        h_x[i] = tx * 40 + 1024;
        h_y[i] = ty * 40 + 512;
        h_z[i] = tz * 40 + 10000.0;
    }
    fclose(fpten);

    // デバイスメモリ割り当て
    float *d_x, *d_y;
    double *d_z, *d_kekka;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMalloc((void**)&d_z, n * sizeof(double));
    cudaMalloc((void**)&d_kekka, yoko * tate * sizeof(double));

    // ホストからデバイスへデータを転送
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_kekka, 0, yoko * tate * sizeof(double));

    // 計算に使用する定数
    double k = (2.0 * 3.14159265358979323846) / (633.0 * pow(10.0, -9.0));
    double p = 10.5 * pow(10.0, -6.0);
    double K = k * p;

    // カーネル呼び出し
    dim3 blockDim(4,4);
    //dim3 gridDim((yoko + blockDim.x - 1) / blockDim.x, (tate + blockDim.y - 1) / blockDim.y);
    int gridX = (yoko + blockDim.x - 1) / blockDim.x;
    int gridY = (tate + blockDim.y - 1) / blockDim.y;
    dim3 gridDim(gridX, gridY);

    printf(" block suu : (%d, %d)\n", gridDim.x, gridDim.y);
    printf("thread suu : (%d, %d)\n", blockDim.x, blockDim.y);

    //clock_t start_time=clock();
    keisan<<<gridDim, blockDim>>>(d_x, d_y, d_z, d_kekka, n, K);
    cudaDeviceSynchronize();
    //clock_t end_time=clock();

    // デバイスからホストへ結果を転送
    cudaMemcpy(kekka, d_kekka, yoko * tate * sizeof(double), cudaMemcpyDeviceToHost);

    // 結果処理
    for (i = 0; i < yoko; i++) {
        for (j = 0; j < tate; j++) {
            if (kekka[i][j] > max_kekka) max_kekka = kekka[i][j];
            if (kekka[i][j] < min_kekka) min_kekka = kekka[i][j];
        }
    }
    mid_kekka = (max_kekka + min_kekka) / 2;

    for (i = 0; i < yoko; i++) {
        for (j = 0; j < tate; j++) {
            if (kekka[i][j] > mid_kekka) img[i][j] = 255;
            else img[i][j] = 0;
        }
    }

    // 結果をファイルに保存
    FILE *fout = fopen("chessdouble.bmp", "wb");
    BITMAPFILEHEADER fileh;
    BITMAPINFOHEADER infoh;
    RGBQUAD rgb[256];

    fileh.bfType = 19778;
    fileh.bfSize = 14 + 40 + 1024 + (yoko * tate);
    fileh.bfReserved1 = 0;
    fileh.bfReserved2 = 0;
    fileh.bf0ffBits = 14 + 40 + 1024;

    infoh.biSize = 40;
    infoh.biWidth = yoko;
    infoh.biHeight = tate;
    infoh.biPlanes = 1;
    infoh.biBitCount = 8;
    infoh.biCompression = 0;
    infoh.biSizeImage = 0;
    infoh.biXPelsPerMeter = 0;
    infoh.biYPelsPerMeter = 0;
    infoh.biCirUsed = 0;
    infoh.biCirImportant = 0;

    fwrite(&fileh, sizeof(fileh), 1, fout);
    fwrite(&infoh, sizeof(infoh), 1, fout);
    for (int k = 0; k < 256; k++) {
        rgb[k].rgbBlue = k;
        rgb[k].rgbGreen = k;
        rgb[k].rgbRed = k;
        rgb[k].rgbReserved = 0;
    }
    fwrite(rgb, sizeof(RGBQUAD), 256, fout);

    for (j = 0; j < tate; j++) {
        for (i = 0; i < yoko; i++) {
            fwrite(&img[i][j], sizeof(unsigned char), 1, fout);
        }
    }
    fclose(fout);



	//double elapsed_time=(double)(end_time-start_time);
	//elapsed_time*=0.000001;

	//printf("Calculation time is %f s\n", elapsed_time);

    // メモリ解放
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_kekka);
    free(h_x);
    free(h_y);
    free(h_z);

    return 0;
}
