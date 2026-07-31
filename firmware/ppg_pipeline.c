#include "ppg_pipeline.h"
#include "cvapp_mb_cls.h"
#include "tensorflow/lite/c/common.h"
#include "xprintf.h"
#include <math.h>
#include <string.h>

#define CPU_CLK 0xffffff+1   // = 16.777.216
#define WINDOW_SIZE 1000
#define NUM_SECTIONS 4

/***************************************************************************
 * BANDPASS FILTER (biquad, koefisien dari scipy butter(4,[0.5,8],fs=100)) *
 ***************************************************************************/
typedef struct {
    float b0, b1, b2, a1, a2;
    float z1, z2;
} biquad_t;

static biquad_t bandpass_sections[NUM_SECTIONS] = {
    {0.00178261f, 0.00356522f, 0.00178261f, -1.29176642f, 0.43571758f, 0.0f, 0.0f},
    {1.0f,        2.0f,        1.0f,        -1.51260719f, 0.71952409f, 0.0f, 0.0f},
    {1.0f,       -2.0f,        1.0f,        -1.93751823f, 0.93871307f, 0.0f, 0.0f},
    {1.0f,       -2.0f,        1.0f,        -1.97736736f, 0.97837261f, 0.0f, 0.0f},
};

static float biquad_process(biquad_t *bq, float input) {
    float output = bq->b0 * input + bq->z1;
    bq->z1 = bq->b1 * input - bq->a1 * output + bq->z2;
    bq->z2 = bq->b2 * input - bq->a2 * output;
    return output;
}

static float bandpass_filter_sample(float raw_sample) {
    float x = raw_sample;
    for (int i = 0; i < NUM_SECTIONS; i++) {
        x = biquad_process(&bandpass_sections[i], x);
    }
    return x;
}

/*****************
 * BUFFER WINDOW *
 *****************/
static float ppg_buffer[WINDOW_SIZE];
static int buffer_index = 0;

/********
 * INIT *
 ********/
void ppg_pipeline_init(void) {
    buffer_index = 0;
    for (int i = 0; i < NUM_SECTIONS; i++) {
        bandpass_sections[i].z1 = 0.0f;
        bandpass_sections[i].z2 = 0.0f;
    }
    xprintf("PPG pipeline initialized (window=%d samples)\n", WINDOW_SIZE);
}

/************************************************************************
 * PROSES 1 SAMPLE BARU (dipanggil tiap kali ada data baru dari sensor) *
 ************************************************************************/
void ppg_pipeline_process_sample(float raw_sample) {
    float filtered = bandpass_filter_sample(raw_sample);
    ppg_buffer[buffer_index++] = filtered;

    if (buffer_index >= WINDOW_SIZE) {
        /*** Normalisasi z-score ***/
        float mean = 0.0f, std = 0.0f;
        for (int i = 0; i < WINDOW_SIZE; i++) mean += ppg_buffer[i];
        mean /= WINDOW_SIZE;

        for (int i = 0; i < WINDOW_SIZE; i++) {
            float diff = ppg_buffer[i] - mean;
            std += diff * diff;
        }
        std = sqrtf(std / WINDOW_SIZE) + 1e-8f;

        /*** Ambil scale & zero_point dari tensor input model ***/
        float input_scale = ((TfLiteAffineQuantization*)(mb_cls_input->quantization.params))->scale->data[0];
        int input_zero_point = ((TfLiteAffineQuantization*)(mb_cls_input->quantization.params))->zero_point->data[0];

        /*** Normalisasi + kuantisasi, langsung isi ke tensor input model ***/
        for (int i = 0; i < WINDOW_SIZE; i++) {
            float normalized = (ppg_buffer[i] - mean) / std;
            int q = (int)roundf(normalized / input_scale + input_zero_point);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            mb_cls_input->data.int8[i] = (int8_t)q;
        }

	/*** Inferensi ***/
        uint32_t tick1, tick2, loop1, loop2;
        SystemGetTick(&tick1, &loop1);
	struct_yolov8_ob_algoResult af_result;
	cv_mb_cls_infer(&af_result);
        SystemGetTick(&tick2, &loop2);
	uint32_t cycles = (loop2 - loop1) * CPU_CLK + (tick1 - tick2);

	/****************************************
         * KIRIM DATA KE SERIAL UNTUK DASHBOARD *
         ****************************************/

	/*** Kirim Data Normalized ***/
        xprintf("DATA:");
        for (int i = 0; i < WINDOW_SIZE; i++) {
            float normalized = (ppg_buffer[i] - mean) / std;
            int norm_scaled = (int)(normalized * 1000);
            xprintf(" %d", norm_scaled);
        }
        xprintf("\n");

        /*** Kirim prediksi (0 = non-AF, 1 = AF) ***/
        xprintf("PRED: %d\n", af_result.obr[0].class_idx);
        /*** Kirim confidence (persen) ***/
        xprintf("CONF: %d\n", (int)(af_result.obr[0].confidence * 100));
        /*** Kirim inference time (cycles) ***/
        xprintf("TIME: %d\n", (int)cycles);

        xprintf("Inferensi selesai. Class: %d, Confidence: %d%%\n",
                af_result.obr[0].class_idx, (int)(af_result.obr[0].confidence * 100));

        buffer_index = 0;  // reset, mulai window berikutnya
	xprintf("\n============================================================\n\n");
    }
}
