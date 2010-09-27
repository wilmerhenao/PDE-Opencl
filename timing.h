#if !defined(TIMING_H_)
#define TIMING_H_ 1

#include <time.h>

#if defined(__cplusplus)
extern "C" {
#endif

long double timespec_to_dbl (struct timespec x);
long double timespec_diff (struct timespec start, struct timespec finish);

long double timer_resolution (void);
void get_time (struct timespec*);

#if defined(__cplusplus)
}
#endif

#endif // TIMING_H_
