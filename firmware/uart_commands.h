#ifndef UART_COMMANDS_H
#define UART_COMMANDS_H

#define MODE_SENSOR  0
#define MODE_GOLDEN  1

void uart_check_and_process(void);
void send_golden_data(void);
void uart_init(void);

#endif
