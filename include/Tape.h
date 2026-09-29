#include "1dValue.h"

typedef struct Tape {
    Value **slots;
    unsigned long long count;
} Tape;

Tape *create_tape(void);

void add_value(Tape *t, Value *v);

Value *sum_values(Tape *t, Value *v1, Value *v2);

Value *mul_values(Tape *t, Value *v1, Value *v2);

Value *sub_values(Tape *t, Value *v1, Value *v2);

Value *tanh_value(Tape *t, Value *v);

void tape_backward(Tape *t);

void tape_reset(Tape *t);

void tape_update(Tape *t);