

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

/** \brief number of nano seconds in a second */
#define NS_PER_SECOND 1000000000;

/** \brief number of threads to use */
#define nWorkers 8

/** \brief threads return determinant array */
double * determinantResults;

/** \brief mutex used for the threads syncronization */
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/** \brief matrix index for concurrent looping */
int matrix_idx = 0;

/** \brief number of matrices inside the file */
int number_of_matrices = 0;

/** \brief order of the matrices */
int order;


void print_matrix(double matrix[][order]) {
    printf("\n");
    for(int i=0; i<order; i++) {
        for(int j=0; j<order; j++) {
            printf("%f  ", matrix[i][j]);
        }
        printf("\n");
    }
}

void switch_row(double matrix[][order], int row1, int row2) {
    double aux;
    for(int i=0;i<order;i++) {
        aux = matrix[row1][i];
        matrix[row1][i] = matrix[row2][i];
        matrix[row2][i] = aux;
    }
}

double compute_determinant(double matrix[][order]) {
    int sign = 1;
    double ratio, determinant = 1;
    
    for(int i=0;i<order;i++) {
        // check if the row can be used, otherwise, switch that row
        if(matrix[i][i] == 0) {
            for(int j=i+1;j<order;j++) {
                if(matrix[j][i] != 0) {
                    switch_row(matrix, i, j);
                    sign = (sign == 1) ? -1: 1;
                    break;
                }
            }                
        }

        for(int j=i+1;j<order;j++) {
            ratio = matrix[j][i]/matrix[i][i];
            for(int k=0;k<order;k++) {
                matrix[j][k] = matrix[j][k]-ratio*matrix[i][k];
            }
        }
        determinant *= matrix[i][i];
    }

    return determinant * sign;
}

void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
{
    td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    td->tv_sec  = t2.tv_sec - t1.tv_sec;
    if (td->tv_sec > 0 && td->tv_nsec < 0)
    {
        td->tv_nsec += NS_PER_SECOND;
        td->tv_sec--;
    }
    else if (td->tv_sec < 0 && td->tv_nsec > 0)
    {
        td->tv_nsec -= NS_PER_SECOND;
        td->tv_sec++;
    }
}

struct timespec time_passed;


void * compute_determinant_thread(void * argv) {
    FILE * ptrFile = (FILE *)argv;
    int idx = 0;
    double matrix[order][order];
    double determinant = 0;

    while(idx < number_of_matrices) {
        // using mutex to preserve order
        pthread_mutex_lock(&mutex);
        idx = matrix_idx++;
        // read matrix from file

        struct timespec start_time_reading, stop_time_reading, elapsed_time_reading;
        clock_gettime(CLOCK_MONOTONIC, &start_time_reading);

        fread(&matrix, sizeof(double), order*order, ptrFile);

        clock_gettime(CLOCK_MONOTONIC, &stop_time_reading);
        sub_timespec(start_time_reading, stop_time_reading, &elapsed_time_reading);

        time_passed.tv_nsec += elapsed_time_reading.tv_nsec;
        time_passed.tv_sec += elapsed_time_reading.tv_sec;



        pthread_mutex_unlock(&mutex);

        // printf("Processing Matrix %d\n", idx);

        // compute determinant
        determinantResults[idx] = compute_determinant(matrix);
        // printf("Computed matrix %d\n", idx);
    }

    return NULL;
}


int compute_determinats_in_file(FILE * ptrFile) {

    // get number of matrices and the order of the matrices
    fread(&number_of_matrices, sizeof(int), 1, ptrFile);
    fread(&order, sizeof(int), 1, ptrFile);

    determinantResults = malloc(sizeof(double)*number_of_matrices);

    // thread initialization
    pthread_t threadWorkers[nWorkers];

    // create concurrent threads
    for(int i=0;i<sizeof(threadWorkers)/sizeof(pthread_t);i++)
        pthread_create(&threadWorkers[i], NULL, compute_determinant_thread, (void *)ptrFile);

    // finish threads
    for(int i=0;i<sizeof(threadWorkers)/sizeof(pthread_t);i++)
        pthread_join(threadWorkers[i], NULL);

    return EXIT_SUCCESS;
}

void test() {
    order = 3;
    double matrix[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    
    double v = compute_determinant(matrix);
    print_matrix(matrix);
    printf("det: %f", v);
}

int main(int argc, char *argv[]) {

    int opt;
    char * file_name;

    do {
        switch((opt = getopt(argc, argv, "f:h"))) {
            case 'f':
                file_name = optarg;
                break;
                
            case 'h':
                printf("-f      --- filename\n");
                break;
        }
    }
    while(opt != -1);

    // open file
    FILE * ptrFile = fopen(optarg, "r");

    // check if file exists
    if(ptrFile == NULL) {
        perror("Error opening file");
        printf("%s\n",optarg);
        return EXIT_FAILURE;
    }

    // initialize time variables and start clock
    struct timespec start_time, stop_time, elapsed_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // compute determinants
    compute_determinats_in_file(ptrFile);

    // stop clock and compute elapsed time
    clock_gettime(CLOCK_MONOTONIC, &stop_time);
    sub_timespec(start_time, stop_time, &elapsed_time);

    // print results
    for(int i=0;i<number_of_matrices;i++)
        printf("The determinant of matrix %d is %.3e\n", i+1, determinantResults[i]);

    printf("\nElapsed time: %d.%.9ld s\n", (int)elapsed_time.tv_sec, elapsed_time.tv_nsec);

    printf("\nElapsed time reading: %d.%.9ld s\n", (int)time_passed.tv_sec, time_passed.tv_nsec);


    // close file
    fclose(ptrFile);
    
    return EXIT_SUCCESS;
}