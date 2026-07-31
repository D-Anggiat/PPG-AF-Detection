/*
 * cvapp_af_detection.cpp
 * Adaptasi untuk model AF Detection (1D-CNN, input PPG 1000 sample)
 */

#include <cstdio>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "WE2_device.h"
#include "board.h"
#include "cvapp_mb_cls.h"
#include "cisdp_sensor.h"
#include "WE2_core.h"
#include "ethosu_driver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif
#include "xprintf.h"
#include "spi_master_protocol.h"
#include "cisdp_cfg.h"
#include "memory_manage.h"
#include <send_result.h>
#include <forward_list>
#include "golden_data.h"   

#define AF_INPUT_TENSOR_LENGTH   1000
#define AF_INPUT_TENSOR_CHANNEL  1
#define AF_NUM_CLASSES           2

#define MB_CLS_DBG_APP_LOG 0

#define TOTAL_STEP_TICK

uint32_t systick_1, systick_2;
uint32_t loop_cnt_1, loop_cnt_2;
#define CPU_CLK 0xffffff+1
static uint32_t capture_image_tick = 0;
#ifdef TRUSTZONE_SEC
#define U55_BASE        BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE        BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE        BASE_ADDR_APB_U55_CTRL
#endif
#endif

using namespace std;

namespace {
constexpr int tensor_arena_size = 450*1024;
static uint32_t tensor_arena=0;
struct ethosu_driver ethosu_drv;
tflite::MicroInterpreter *mb_cls_int_ptr=nullptr;
};

TfLiteTensor *mb_cls_input = nullptr;
TfLiteTensor *mb_cls_output = nullptr;

static void _arm_npu_irq_handler(void)
{
    ethosu_irq_handler(&ethosu_drv);
}

static void _arm_npu_irq_init(void)
{
    const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;
    EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);
    NVIC_EnableIRQ(ethosu_irqnum);
}

static int _arm_npu_init(bool security_enable, bool privilege_enable)
{
    int err = 0;
    _arm_npu_irq_init();
#if TFLM2209_U55TAG2205
    const void * ethosu_base_address = (void *)(U55_BASE);
#else
    void * const ethosu_base_address = (void *)(U55_BASE);
#endif
    if (0 != (err = ethosu_init(&ethosu_drv, ethosu_base_address, NULL, 0,
                                security_enable, privilege_enable))) {
        xprintf("failed to initalise Ethos-U device\n");
        return err;
    }
    xprintf("Ethos-U55 device initialised\n");
    return 0;
}

int cv_mb_cls_init(bool security_enable, bool privilege_enable, uint32_t model_addr) {
    int ercode = 0;
    tensor_arena = mm_reserve_align(tensor_arena_size, 0x20);
    xprintf("TA[%x]\r\n", tensor_arena);

    if (_arm_npu_init(security_enable, privilege_enable) != 0)
        return -1;

    if (model_addr != 0) {
        static const tflite::Model* af_model = tflite::GetModel((const void *)model_addr);

        if (af_model->version() != TFLITE_SCHEMA_VERSION) {
            xprintf("[ERROR] af_model's schema version %d != supported %d\n",
                    af_model->version(), TFLITE_SCHEMA_VERSION);
            return -1;
        }
        xprintf("af model's schema version %d\n", af_model->version());

        static tflite::MicroMutableOpResolver<1> af_op_resolver;
        if (kTfLiteOk != af_op_resolver.AddEthosU()) {
            xprintf("Failed to add Arm NPU support to op resolver.");
            return false;
        }

        static tflite::MicroInterpreter af_static_interpreter(af_model, af_op_resolver,
                        (uint8_t*)tensor_arena, tensor_arena_size);

        if (af_static_interpreter.AllocateTensors() != kTfLiteOk) {
            xprintf("AllocateTensors failed\n");
            return false;
        }
        mb_cls_int_ptr = &af_static_interpreter;
        mb_cls_input = af_static_interpreter.input(0);
        mb_cls_output = af_static_interpreter.output(0);

        xprintf("Input bytes: %d | Output bytes: %d\n",
                mb_cls_input->bytes, mb_cls_output->bytes);
    }

    xprintf("initial done\n");
    return ercode;
}

int cv_mb_cls_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob) {
    int ercode = 0;

    if (mb_cls_int_ptr != nullptr) {
        SystemGetTick(&systick_1, &loop_cnt_1);

        xprintf("=== GOLDEN CHECK (AF Detection) ===\n");
        xprintf("Input tensor bytes: %d\n", mb_cls_input->bytes);

        if (mb_cls_input->bytes != sizeof(golden_input)) {
            xprintf("WARNING: ukuran golden_input (%d) tidak cocok dengan input tensor (%d)\n",
                     sizeof(golden_input), mb_cls_input->bytes);
        }

        memcpy(mb_cls_input->data.int8, golden_input, mb_cls_input->bytes);

        TfLiteStatus invoke_status = mb_cls_int_ptr->Invoke();

        SystemGetTick(&systick_2, &loop_cnt_2);

        if (invoke_status != kTfLiteOk) {
            xprintf("af detection invoke fail\n");
            return -1;
        }

        xprintf("Output tensor bytes: %d\n", mb_cls_output->bytes);

        int af_best = 0;
        for (int i = 0; i < mb_cls_output->bytes; ++i) {
            int val = (int)mb_cls_output->data.int8[i];
            xprintf("out[%d] = %d (golden = %d)\n", i, val, (int)golden_output[i]);
            if (mb_cls_output->data.int8[i] > mb_cls_output->data.int8[af_best]) {
                af_best = i;
            }
        }

        const char* af_labels[] = {"non-AF", "AF"};
        xprintf("Predicted class: %d (%s)\n", af_best, af_labels[af_best]);

        float output_scale = ((TfLiteAffineQuantization*)(mb_cls_output->quantization.params))->scale->data[0];
        int output_zeropoint = ((TfLiteAffineQuantization*)(mb_cls_output->quantization.params))->zero_point->data[0];
        algoresult_yolov8n_ob->obr[0].confidence =
            ((float)mb_cls_output->data.int8[af_best] - (float)output_zeropoint) * output_scale;
        algoresult_yolov8n_ob->obr[0].class_idx = af_best;
    }

    SystemGetTick(&systick_1, &loop_cnt_1);
    SystemGetTick(&systick_2, &loop_cnt_2);
    capture_image_tick = (loop_cnt_2-loop_cnt_1)*CPU_CLK+(systick_1-systick_2);
    return ercode;
}

/********************
 * FUNGSI INFERENSI *
 ********************/
int cv_mb_cls_infer(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob) {
    if (mb_cls_int_ptr == nullptr) {
        xprintf("Error: interpreter not initialized\n");
        return -1;
    }

    uint32_t systick_1, systick_2;
    uint32_t loop_cnt_1, loop_cnt_2;
    SystemGetTick(&systick_1, &loop_cnt_1);

    TfLiteStatus invoke_status = mb_cls_int_ptr->Invoke();

    SystemGetTick(&systick_2, &loop_cnt_2);

    if (invoke_status != kTfLiteOk) {
        xprintf("Inferensi gagal!\n");
        return -1;
    }

    xprintf("Output tensor bytes: %d\n", mb_cls_output->bytes);

    int af_best = 0;
    for (int i = 0; i < mb_cls_output->bytes; ++i) {
        int val = (int)mb_cls_output->data.int8[i];
        xprintf("out[%d] = %d\n", i, val);
        if (mb_cls_output->data.int8[i] > mb_cls_output->data.int8[af_best]) {
            af_best = i;
        }
    }

    const char* af_labels[] = {"non-AF", "AF"};
    xprintf("Predicted class: %d (%s)\n", af_best, af_labels[af_best]);
    xprintf("Inference time: %d cycles\n", (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2));
 
    if (algoresult_yolov8n_ob) {
        float output_scale = ((TfLiteAffineQuantization*)(mb_cls_output->quantization.params))->scale->data[0];
        int output_zeropoint = ((TfLiteAffineQuantization*)(mb_cls_output->quantization.params))->zero_point->data[0];
        algoresult_yolov8n_ob->obr[0].confidence =
            ((float)mb_cls_output->data.int8[af_best] - (float)output_zeropoint) * output_scale;
        algoresult_yolov8n_ob->obr[0].class_idx = af_best;
    }

    return 0;
}

int cv_mb_cls_deinit()
{
    return 0;
}
