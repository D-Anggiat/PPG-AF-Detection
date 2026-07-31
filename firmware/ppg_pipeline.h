#ifndef PPG_PIPELINE_H
#define PPG_PIPELINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void ppg_pipeline_init(void);
void ppg_pipeline_process_sample(float raw_sample);

#ifdef __cplusplus
}
#endif

#endif
