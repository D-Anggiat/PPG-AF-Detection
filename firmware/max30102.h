#ifndef MAX30102_H
#define MAX30102_H

#include <stdint.h>
#include "hx_drv_iic.h"

#ifdef __cplusplus
extern "C" {
#endif

// I2C address (7-bit)
#define MAX30102_I2C_ADDR       0x57

// Register map
#define MAX30102_INTR_STATUS_1  0x00
#define MAX30102_INTR_STATUS_2  0x01
#define MAX30102_INTR_ENABLE_1  0x02
#define MAX30102_INTR_ENABLE_2  0x03
#define MAX30102_FIFO_WR_PTR    0x04
#define MAX30102_OVF_COUNTER    0x05
#define MAX30102_FIFO_RD_PTR    0x06
#define MAX30102_FIFO_DATA      0x07
#define MAX30102_FIFO_CONFIG    0x08
#define MAX30102_MODE_CONFIG    0x09
#define MAX30102_SPO2_CONFIG    0x0A
#define MAX30102_LED1_PA        0x0C
#define MAX30102_LED2_PA        0x0D
#define MAX30102_PART_ID        0xFF

// Expected values
#define MAX30102_PART_ID_VALUE  0x15

// Config values (HR-only mode)
#define MAX30102_MODE_HR_ONLY   0x02   // Heart Rate mode, Red LED only
#define MAX30102_MODE_RESET     0x40   // Reset bit in MODE_CONFIG

// SPO2_CONFIG: ADC range 4096nA, sample rate 100Hz, pulse width 411us
// bit 6:5 SPO2_ADC_RGE = 01 (4096nA)
// bit 4:2 SPO2_SR      = 001 (100 Hz)
// bit 1:0 LED_PW       = 11 (411 us, 18-bit ADC)
#define MAX30102_SPO2_CFG       0x27

// FIFO_CONFIG: sample averaging = 4, FIFO rollover enable
// bit 7:5 SMP_AVE = 010 (4 samples averaged)
// bit 4   FIFO_ROLLOVER_EN = 1
// bit 3:0 FIFO_A_FULL = 0000
#define MAX30102_FIFO_CFG       0x10

// LED current (0x1F ~ 6.2 mA — safe, avoid overheating)
#define MAX30102_LED_CURRENT_LOW  0x1F

// Public API
int max30102_init(void);
int max30102_read_hr_sample(uint32_t *out_red);
int max30102_available_samples(uint8_t *out_count);

#ifdef __cplusplus
}
#endif

#endif /* MAX30102_H */
