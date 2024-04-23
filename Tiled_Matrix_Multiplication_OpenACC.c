#include <stdio.h>
#include <stdlib.h>

#define row1 2
#define column1 2
#define row2 2
#define column2 3
#define TILE_DIM 1 

void multiplyMatrixTiled(int m1[][column1], int m2[][column2]) {
    int result[row1][column2];

    printf("Tiled Resultant Matrix is:\n");

    #pragma acc data copyin(m1, m2) copyout(result)
    {
        #pragma acc parallel loop tile(TILE_DIM, TILE_DIM)
        for (int i = 0; i < row1; i++) {
            for (int j = 0; j < column2; j++) {
                int temp = 0;
                for (int k = 0; k < row2; k++) {
                    temp += m1[i][k] * m2[k][j];
                }
                result[i][j] = temp;
                printf("%d\t", result[i][j]);
            }
            printf("\n");
        }
    }
}

int main() {
    int m1[row1][column1] = { { 1, 1 }, { 2, 2 } };
    int m2[row2][column2] = { { 1, 1, 1 }, { 2, 2, 2 } };

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    multiplyMatrixTiled(m1, m2);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9; 

    return 0;
}
