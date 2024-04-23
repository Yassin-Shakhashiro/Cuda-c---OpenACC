#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define row1 2 
#define column1 2 
#define row2 2 
#define column2 3 

void multiplyMatrix(int m1[][column1], int m2[][column2])
{
	int result[row1][column2];

	printf("Resultant Matrix is:\n");

	for (int i = 0; i < row1; i++) {
		for (int j = 0; j < column2; j++) {
			result[i][j] = 0;

			for (int k = 0; k < row2; k++) {
				result[i][j] += m1[i][k] * m2[k][j];
			}

			printf("%d\t", result[i][j]);
		}

		printf("\n");
	}
}

// Driver code
int main()
{
	int m1[row1][column1] = { { 1, 1 }, { 2, 2 } };

	int m2[row2][column2] = { { 1, 1, 1 }, { 2, 2, 2 } };

    clock_t t;
    t = clock();

    multiplyMatrix(m1, m2);

    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
	
    printf("%lf",time_taken);
	return 0;
}
