# STUNIR Embedded Makefile: module
# Architecture: arm

PREFIX = arm-none-eabi-
CC = $(PREFIX)gcc
OBJCOPY = $(PREFIX)objcopy
SIZE = $(PREFIX)size

CFLAGS = -mcpu=cortex-m3 -mthumb -Os -Wall -fno-common
CFLAGS += -ffunction-sections -fdata-sections
CFLAGS += -DSTUNIR_EMBEDDED -I.

LDFLAGS = -Tmodule.ld -nostartfiles -Wl,--gc-sections

SRCS = module.c startup.c
OBJS = $(SRCS:.c=.o)
TARGET = module

.PHONY: all clean size

all: $(TARGET).elf $(TARGET).bin $(TARGET).hex size

$(TARGET).elf: $(OBJS)
        $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

$(TARGET).bin: $(TARGET).elf
        $(OBJCOPY) -O binary $< $@

$(TARGET).hex: $(TARGET).elf
        $(OBJCOPY) -O ihex $< $@

%.o: %.c
        $(CC) $(CFLAGS) -c -o $@ $<

size: $(TARGET).elf
        $(SIZE) $<

clean:
        rm -f $(OBJS) $(TARGET).elf $(TARGET).bin $(TARGET).hex
