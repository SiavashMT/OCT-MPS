#include "rng.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int InitRNG(unsigned long long *x, unsigned int *a, unsigned long long *xR, unsigned int *aR, unsigned long long *xS,
            unsigned int *aS, const unsigned int n_rng, const char *safeprimes_file,
            unsigned long long xinit) {
    // xinit is the SEED (given or taken from clock)
    FILE *fp;
    unsigned int begin = 0u, beginR = 0u, beginS = 0u;
    unsigned int fora, forR, forS, tmp1, tmp2;
    unsigned long long xinitR = SEED1, xinitS = SEED2;  // xinit=SEED0;

    if (strlen(safeprimes_file) == 0) {
        // Try to find it in the local directory
        safeprimes_file = "safeprimes_base32.txt";
    }

    fp = fopen(safeprimes_file, "r");

    if (fp == NULL) {
        printf("Could not find the file of safeprimes (%s)! Terminating!\n",
               safeprimes_file);
        return EXIT_FAILURE;
    }

    fscanf(fp, "%u %u %u", &begin, &tmp1, &tmp2);   // first safe prime
    fscanf(fp, "%u %u %u", &beginR, &tmp1, &tmp2);  // safe prime for ReflectT
    fscanf(fp, "%u %u %u", &beginS, &tmp1, &tmp2);  // safe prime for Spin

    // Here we set up a loop, using the first multiplier in the file to
    // generate x's and c's
    // There are some restrictions to these two numbers:
    // 0<=c<a and 0<=x<b, where a is the multiplier and b is the base (2^32)
    // also [x,c]=[0,0] and [b-1,a-1] are not allowed.

    // Make sure xinit is a valid seed (using the above mentioned
    // restrictions) HINT: xinit is the SEED (given or taken from clock)
    if ((xinit == 0ull) | (((unsigned int)(xinit >> 32)) >= (begin - 1))
            | (((unsigned int) xinit) >= 0xfffffffful)) {
        // xinit (probably) not a valid seed!
        //(we have excluded a few unlikely exceptions)
        printf("%llu not a valid seed! Terminating!\n", xinit);
        return EXIT_FAILURE;
    }
    // initial declarations are only allowed in C99 or C11 mode
    unsigned int i;
    for (i = 0; i < n_rng; i++) {
        fscanf(fp, "%u %u %u", &fora, &tmp1, &tmp2);
        a[i] = fora;
        x[i] = 0;

        // seed for ReflectTransmit
        fscanf(fp, "%u %u %u", &forR, &tmp1, &tmp2);
        aR[i] = forR;
        xR[i] = 0;

        // seed for Spin
        fscanf(fp, "%u %u %u", &forS, &tmp1, &tmp2);
        aS[i] = forS;
        xS[i] = 0;

        while ((x[i] == 0) | (((unsigned int)(x[i] >> 32)) >= (fora - 1))
                | (((unsigned int) x[i]) >= 0xfffffffful)) {
            // generate a random number
            // HINT: xinit is the SEED (given or taken from clock) and begin is
            // the first safe prime in the list
            xinit = (xinit & 0xffffffffull) * (begin) + (xinit >> 32);

            // calculate c and store in the upper 32 bits of x[i]
            x[i] = (unsigned int) floor(
                       (((double) ((unsigned int) xinit)) / (double) 0x100000000)
                       * fora);  // Make sure 0<=c<a
            x[i] = x[i] << 32;

            // generate a random number and store in the lower 32 bits of x[i]
            //(as the initial x of the generator)
            // x will be 0<=x<b, where b is the base 2^32
            xinit = (xinit & 0xffffffffull) * (begin) + (xinit >> 32);
            x[i] += (unsigned int) xinit;
        }  // End while x[i]

        while ((xR[i] == 0) | (((unsigned int)(xR[i] >> 32)) >= (forR - 1))
                | (((unsigned int) xR[i]) >= 0xfffffffful)) {
            // generate a random number
            // HINT: xinit is the SEED (given or taken from clock) and begin is
            // the first safe prime in the list
            xinitR = (xinitR & 0xffffffffull) * (beginR) + (xinitR >> 32);

            // calculate c and store in the upper 32 bits of x[i]
            xR[i] = (unsigned int) floor(
                        (((double) ((unsigned int) xinitR)) / (double) 0x100000000)
                        * forR);  // Make sure 0<=c<a
            xR[i] = xR[i] << 32;

            // generate a random number and store in the lower 32 bits of x[i]
            //(as the initial x of the generator)
            // x will be 0<=x<b, where b is the base 2^32
            xinitR = (xinitR & 0xffffffffull) * (beginR) + (xinitR >> 32);
            xR[i] += (unsigned int) xinitR;
        }  // End while x[i]

        while ((xS[i] == 0) | (((unsigned int)(xS[i] >> 32)) >= (forS - 1))
                | (((unsigned int) xS[i]) >= 0xfffffffful)) {
            // generate a random number
            // HINT: xinit is the SEED (given or taken from clock) and begin is
            // the first safe prime in the list
            xinitS = (xinitS & 0xffffffffull) * (beginS) + (xinitS >> 32);

            // calculate c and store in the upper 32 bits of x[i]
            xS[i] = (unsigned int) floor(
                        (((double) ((unsigned int) xinitS)) / (double) 0x100000000)
                        * forS);  // Make sure 0<=c<a
            xS[i] = xS[i] << 32;

            // generate a random number and store in the lower 32 bits of x[i]
            //(as the initial x of the generator)
            // x will be 0<=x<b, where b is the base 2^32
            xinitS = (xinitS & 0xffffffffull) * (beginS) + (xinitS >> 32);
            xS[i] += (unsigned int) xinitS;
        }
    }
    fclose(fp);

    return EXIT_SUCCESS;
}
