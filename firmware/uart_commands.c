#include "uart_commands.h"
#include "xprintf.h"
#include <string.h>
#include <stdlib.h>
#include "hx_drv_uart.h"
#include "golden_all.h"
#include "cvapp_mb_cls.h"
#include "WE2_device.h"
#define CPU_CLK 0xffffff+1

extern uint8_t g_mode;
extern uint8_t g_patient_idx;
extern uint8_t g_window_idx;
extern uint8_t g_golden_active;

static struct dev_uart* g_uart_dev = NULL;

/*** Inisialisasi UART ***/
void uart_init(void) {
    if (g_uart_dev == NULL) {
        g_uart_dev = hx_drv_uart_get_dev(USE_DW_UART_0);
        if (g_uart_dev == NULL) {
            xprintf("UART init failed\n");
        } else {
            xprintf("UART initialized\n");
        }
    }
}

void uart_check_and_process(void) {
    if (g_uart_dev == NULL) return;

    uint8_t ch;

    int32_t ret = g_uart_dev->uart_read_nonblock(&ch, 1);
    if (ret != 1) return;  

    char c = (char)ch;

    if (c == 's') {
        g_mode = MODE_SENSOR;
        g_golden_active = 0;
        xprintf("Mode: SENSOR\n");
    }
    else if (c == 'g') {
        g_mode = MODE_GOLDEN;
        g_golden_active = 1;
        xprintf("Mode: GOLDEN\n");
    }
    else if (c == 'n' && g_mode == MODE_GOLDEN) {
        if (g_window_idx < MAX_WINDOWS - 1) {
            g_window_idx++;
            xprintf("Window: %d\n", g_window_idx);
	    send_golden_data(); 
        }
    }
    else if (c == 'p' && g_mode == MODE_GOLDEN) {
        if (g_window_idx > 0) {
            g_window_idx--;
            xprintf("Window: %d\n", g_window_idx);
	    send_golden_data();
        }
    }
    else if (c == 'l' && g_mode == MODE_GOLDEN) {
        char buf[4] = {0};
        int i = 0;
        while (i < 3) {
            uint8_t next_ch;
            int ret2 = g_uart_dev->uart_read_nonblock(&next_ch, 1);
            if (ret2 != 1) break;
            if (next_ch >= '0' && next_ch <= '9') {
                buf[i++] = (char)next_ch;
            } else {
                break;
            }
        }
        int patient = atoi(buf);
        if (patient >= 0 && patient < TOTAL_PATIENTS) {
            g_patient_idx = patient;
            g_window_idx = 0;
            xprintf("Loaded patient %d\n", patient);
        } else {
            xprintf("Invalid patient index\n");
	    send_golden_data();
        }
    }
}

/**********************************************
 * Kirim data dari golden_all.h (mode golden) *
 **********************************************/
void send_golden_data(void) {
    const int8_t* input_data = golden_input_all[g_patient_idx][g_window_idx];
    memcpy(mb_cls_input->data.int8, input_data, WINDOW_SIZE);

    /*** Inferensi ***/
    struct_yolov8_ob_algoResult af_result;
    uint32_t tick1, tick2, loop1, loop2;
    SystemGetTick(&tick1, &loop1);
    cv_mb_cls_infer(&af_result);
    SystemGetTick(&tick2, &loop2);
    uint32_t cycles = (loop2 - loop1) * CPU_CLK + (tick1 - tick2);

    xprintf("DATA:");
    for (int i = 0; i < WINDOW_SIZE; i++) {
        xprintf(" %d", input_data[i]);
    }
    xprintf("\n");
    xprintf("PRED: %d\n", af_result.obr[0].class_idx);
    xprintf("CONF: %d\n", (int)(af_result.obr[0].confidence * 100));
    xprintf("TIME: %d\n", (int)cycles);
}
