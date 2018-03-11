#ifndef OCTMPS_INCLUDE_OCTMPS_H_
#define OCTMPS_INCLUDE_OCTMPS_H_

#ifdef __cplusplus
extern "C" {
#endif

// OCTMPS constants
#define ZERO_FP 0.0 // FP_ZERO is defined in math.h
#define FP_ONE  1.0
#define FP_TWO  2.0

#define PI_const 3.14159265359
#define WEIGHT 1E-4
#define CHANCE 0.1

#define SQ(x) ( (x)*(x) )

//NOTE: Single Precision
#define COSNINETYDEG 1.0E-6
#define COSZERO (1.0F - 1.0E-14)

#define NUM_SUBSTEPS_RESOLUTION 6

#ifdef __cplusplus
}
#endif

#endif