/* two.c -- use the 6 basic MPI calls as described in Ian Foster's
            parallel programming book to implement a simple handshake
            between a pair of processes
   -- Jim Otto, 1/21/2013 -- probably needs more comments in-body     */
   #include <stdio.h>
   #include <stdlib.h>
   #include <unistd.h>
   #include "mpi.h"

   #define ROOT  0
   #define OTHER 1

   int main (int argc, char* argv[])
   {
     int rank, nprocs, ierr, i, error=0;
     size_t nbytes;
     char cptr[100];
     int token;              /* pass this token back and forth...*/
     MPI_Status status;

     gethostname(cptr,100);

     ierr = MPI_Init(&argc, &argv);
     if (ierr != MPI_SUCCESS) {
       printf("MPI initialization error\n");
       exit(0);
     }

     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

     if (nprocs != 2) {
       printf("two error: needs exactly 2 processes...\n");
       printf("rank: %d, nprocs: %d\n", rank, nprocs);
       MPI_Finalize();
       exit(0);
     }

     printf("process (rank) %d of %d (on %s)\n", rank, nprocs, cptr);

     if (rank == ROOT) { /* get things started, do timing */
       token = 0;
       MPI_Send(&token, 1, MPI_INT, OTHER, 0, MPI_COMM_WORLD);
       MPI_Recv(&token, 1, MPI_INT, OTHER, 0, MPI_COMM_WORLD, &status);

       if (token == 1) {
           printf("success (process %d)\n", rank);
       } else {
           printf("failure (process %d)\n", rank);
       }
     }
     else { /* silent partner */
       MPI_Recv(&token, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD, &status);
       token++;
       MPI_Send(&token, 1, MPI_INT, ROOT, 0, MPI_COMM_WORLD);
     }

     MPI_Finalize();
     return 0;
   }
