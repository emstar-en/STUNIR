/* STUNIR Generated Code - Interrupt Handling
 * Module: interrupts
 * Architecture: x86_64
 * DO-178C Level A Compliance
 */

#include "types.h"
#include "io.h"
#include "serial.h"
#include "timer.h"
#include "interrupts.h"

#define IDT_ENTRIES         256
#define PIC1_COMMAND        0x20
#define PIC1_DATA           0x21
#define PIC2_COMMAND        0xA0
#define PIC2_DATA           0xA1
#define PIC_EOI             0x20
#define ICW1_INIT           0x10
#define ICW1_ICW4           0x01
#define ICW4_8086           0x01
#define IRQ_BASE            32

/* IDT entry structure (64-bit) */
struct idt_entry {
    u16 offset_low;
    u16 selector;
    u8  ist;
    u8  type_attr;
    u16 offset_mid;
    u32 offset_high;
    u32 zero;
} __attribute__((packed));

/* IDT pointer structure */
struct idt_ptr {
    u16 limit;
    u64 base;
} __attribute__((packed));

/* IDT and pointer */
static struct idt_entry idt[IDT_ENTRIES];
static struct idt_ptr idtp;

/* External ISR stubs from assembly */
extern void isr0(void);
extern void isr1(void);
extern void isr2(void);
extern void isr3(void);
extern void isr4(void);
extern void isr5(void);
extern void isr6(void);
extern void isr7(void);
extern void isr8(void);
extern void isr9(void);
extern void isr10(void);
extern void isr11(void);
extern void isr12(void);
extern void isr13(void);
extern void isr14(void);
extern void isr15(void);
extern void isr16(void);
extern void isr17(void);
extern void isr18(void);
extern void isr19(void);

extern void irq0(void);
extern void irq1(void);
extern void irq2(void);
extern void irq3(void);
extern void irq4(void);
extern void irq5(void);
extern void irq6(void);
extern void irq7(void);
extern void irq8(void);
extern void irq9(void);
extern void irq10(void);
extern void irq11(void);
extern void irq12(void);
extern void irq13(void);
extern void irq14(void);
extern void irq15(void);

extern void syscall_entry(void);

/* Remap the PIC */
static void pic_remap(u8 offset1, u8 offset2) {
    u8 a1, a2;
    
    /* Save masks */
    a1 = inb(PIC1_DATA);
    a2 = inb(PIC2_DATA);
    
    /* Start init sequence */
    outb(PIC1_COMMAND, ICW1_INIT | ICW1_ICW4);
    io_wait();
    outb(PIC2_COMMAND, ICW1_INIT | ICW1_ICW4);
    io_wait();
    
    /* Set vector offsets */
    outb(PIC1_DATA, offset1);
    io_wait();
    outb(PIC2_DATA, offset2);
    io_wait();
    
    /* Set cascade */
    outb(PIC1_DATA, 4);
    io_wait();
    outb(PIC2_DATA, 2);
    io_wait();
    
    /* 8086 mode */
    outb(PIC1_DATA, ICW4_8086);
    io_wait();
    outb(PIC2_DATA, ICW4_8086);
    io_wait();
    
    /* Restore masks */
    outb(PIC1_DATA, a1);
    outb(PIC2_DATA, a2);
}

/* Set an IDT entry */
void idt_set_gate(u8 num, u64 handler, u16 selector, u8 flags) {
    idt[num].offset_low = handler & 0xFFFF;
    idt[num].selector = selector;
    idt[num].ist = 0;
    idt[num].type_attr = flags;
    idt[num].offset_mid = (handler >> 16) & 0xFFFF;
    idt[num].offset_high = (handler >> 32) & 0xFFFFFFFF;
    idt[num].zero = 0;
}

/* Initialize IDT and install handlers */
void idt_init(void) {
    /* Remap PIC */
    pic_remap(IRQ_BASE, IRQ_BASE + 8);
    
    /* Set up IDT pointer */
    idtp.limit = sizeof(idt) - 1;
    idtp.base = (u64)&idt;
    
    /* Clear IDT */
    for (int i = 0; i < IDT_ENTRIES; i++) {
        idt_set_gate(i, 0, 0, 0);
    }
    
    /* Install CPU exception handlers */
    idt_set_gate(0,  (u64)isr0,  0x08, 0x8E);
    idt_set_gate(1,  (u64)isr1,  0x08, 0x8E);
    idt_set_gate(2,  (u64)isr2,  0x08, 0x8E);
    idt_set_gate(3,  (u64)isr3,  0x08, 0x8E);
    idt_set_gate(4,  (u64)isr4,  0x08, 0x8E);
    idt_set_gate(5,  (u64)isr5,  0x08, 0x8E);
    idt_set_gate(6,  (u64)isr6,  0x08, 0x8E);
    idt_set_gate(7,  (u64)isr7,  0x08, 0x8E);
    idt_set_gate(8,  (u64)isr8,  0x08, 0x8E);
    idt_set_gate(9,  (u64)isr9,  0x08, 0x8E);
    idt_set_gate(10, (u64)isr10, 0x08, 0x8E);
    idt_set_gate(11, (u64)isr11, 0x08, 0x8E);
    idt_set_gate(12, (u64)isr12, 0x08, 0x8E);
    idt_set_gate(13, (u64)isr13, 0x08, 0x8E);
    idt_set_gate(14, (u64)isr14, 0x08, 0x8E);
    idt_set_gate(15, (u64)isr15, 0x08, 0x8E);
    idt_set_gate(16, (u64)isr16, 0x08, 0x8E);
    idt_set_gate(17, (u64)isr17, 0x08, 0x8E);
    idt_set_gate(18, (u64)isr18, 0x08, 0x8E);
    idt_set_gate(19, (u64)isr19, 0x08, 0x8E);
    
    /* Install IRQ handlers */
    idt_set_gate(32, (u64)irq0,  0x08, 0x8E);
    idt_set_gate(33, (u64)irq1,  0x08, 0x8E);
    idt_set_gate(34, (u64)irq2,  0x08, 0x8E);
    idt_set_gate(35, (u64)irq3,  0x08, 0x8E);
    idt_set_gate(36, (u64)irq4,  0x08, 0x8E);
    idt_set_gate(37, (u64)irq5,  0x08, 0x8E);
    idt_set_gate(38, (u64)irq6,  0x08, 0x8E);
    idt_set_gate(39, (u64)irq7,  0x08, 0x8E);
    idt_set_gate(40, (u64)irq8,  0x08, 0x8E);
    idt_set_gate(41, (u64)irq9,  0x08, 0x8E);
    idt_set_gate(42, (u64)irq10, 0x08, 0x8E);
    idt_set_gate(43, (u64)irq11, 0x08, 0x8E);
    idt_set_gate(44, (u64)irq12, 0x08, 0x8E);
    idt_set_gate(45, (u64)irq13, 0x08, 0x8E);
    idt_set_gate(46, (u64)irq14, 0x08, 0x8E);
    idt_set_gate(47, (u64)irq15, 0x08, 0x8E);
    
    /* Load IDT */
    __asm__ volatile ("lidt %0" : : "m"(idtp));
    
    /* Enable only timer interrupt (IRQ0) */
    outb(PIC1_DATA, 0xFE);  /* Enable IRQ0 only */
    outb(PIC2_DATA, 0xFF);  /* Disable all IRQs on slave */
}

/* Common IRQ handler dispatch */
void irq_handler(u32 irq_num) {
    /* Handle timer interrupt */
    if (irq_num == 0) {
        timer_handler();
    }
    
    /* Send EOI */
    if (irq_num >= 8) {
        outb(PIC2_COMMAND, PIC_EOI);
    }
    outb(PIC1_COMMAND, PIC_EOI);
}

/* Exception handler */
void exception_handler(u32 int_no) {
    serial_write_string("\n!!! CPU Exception: ");
    serial_write_hex(int_no);
    serial_write_string(" !!!\n");
    
    /* Halt on exception */
    while (1) {
        __asm__ volatile ("cli; hlt");
    }
}
