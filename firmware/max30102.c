#include "max30102.h"
#include "xprintf.h"

// I2C write helper
static IIC_ERR_CODE_E max_write(uint8_t reg, uint8_t val) {
    uint8_t addr[1];
    addr[0] = reg;
    return hx_drv_i2cm_write_data(USE_DW_IIC_0, MAX30102_I2C_ADDR, addr, 1, &val, 1);
}

// I2C read helper (single byte)
static IIC_ERR_CODE_E max_read(uint8_t reg, uint8_t *data, uint8_t len) {
    uint8_t addr[1];
    addr[0] = reg;
    return hx_drv_i2cm_write_restart_read(USE_DW_IIC_0, MAX30102_I2C_ADDR, addr, 1, data, len);
}

int max30102_init(void) {
    uint8_t tmp;

    // Note: I2C bus assumed already initialized by mpu6050_init().
    // If MPU6050 not initialized first, uncomment the line below:
    // hx_drv_i2cm_init(USE_DW_IIC_0, HX_I2C_HOST_MST_0_BASE, DW_IIC_SPEED_FAST);

    // 1. Verify PART_ID (should be 0x15)
    if (max_read(MAX30102_PART_ID, &tmp, 1) != IIC_ERR_OK) {
        xprintf("MAX30102 read PART_ID fail\n");
        return -1;
    }
    if (tmp != MAX30102_PART_ID_VALUE) {
        xprintf("MAX30102 PART_ID mismatch: 0x%02X (expect 0x15)\n", tmp);
        return -1;
    }
    xprintf("MAX30102 PART_ID OK: 0x%02X\n", tmp);

    // 2. Soft reset
    if (max_write(MAX30102_MODE_CONFIG, MAX30102_MODE_RESET) != IIC_ERR_OK) {
        xprintf("MAX30102 reset fail\n");
        return -1;
    }
    // Wait for reset to complete (reset bit auto-clears)
    for (volatile int d = 0; d < 100000; ++d);

    // 3. Set MODE_CONFIG = HR-only (Red LED)
    if (max_write(MAX30102_MODE_CONFIG, MAX30102_MODE_HR_ONLY/*MAX30102_MODE_SPO2*/) != IIC_ERR_OK) {
        xprintf("MAX30102 mode config fail\n");
        return -1;
    }

    // 4. Set SPO2_CONFIG (ADC range, sample rate, pulse width)
    if (max_write(MAX30102_SPO2_CONFIG, MAX30102_SPO2_CFG) != IIC_ERR_OK) {
        xprintf("MAX30102 spo2 config fail\n");
        return -1;
    }

    // 5. Set FIFO_CONFIG (averaging, rollover)
    if (max_write(MAX30102_FIFO_CONFIG, MAX30102_FIFO_CFG) != IIC_ERR_OK) {
        xprintf("MAX30102 fifo config fail\n");
        return -1;
    }

    // 6. Set LED current (low, ~6.2 mA)
    if (max_write(MAX30102_LED1_PA, MAX30102_LED_CURRENT_LOW) != IIC_ERR_OK) {
        xprintf("MAX30102 LED1 current fail\n");
        return -1;
    }

    // 7. Clear FIFO pointers
    max_write(MAX30102_FIFO_WR_PTR, 0);
    max_write(MAX30102_OVF_COUNTER, 0);
    max_write(MAX30102_FIFO_RD_PTR, 0);

    xprintf("MAX30102 init done\n");
    return 0;
}

int max30102_available_samples(uint8_t *out_count) {
    uint8_t wr_ptr, rd_ptr;
    if (max_read(MAX30102_FIFO_WR_PTR, &wr_ptr, 1) != IIC_ERR_OK) return -1;
    if (max_read(MAX30102_FIFO_RD_PTR, &rd_ptr, 1) != IIC_ERR_OK) return -1;
    int count = wr_ptr - rd_ptr;
    if (count < 0) count += 32;
    *out_count = (uint8_t)count;
    return 0;
}

int max30102_read_hr_sample(uint32_t *out_red) {
    uint8_t data[3];
    if (max_read(MAX30102_FIFO_DATA, data, 3) != IIC_ERR_OK) return -1;
    *out_red = ((uint32_t)data[0] << 16) | ((uint32_t)data[1] << 8) | data[2];
    *out_red &= 0x3FFFF;
    return 0;
}
