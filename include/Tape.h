#ifndef FILE_TAPE_H
#define FILE_TAPE_H

#include "1dValue.h"

typedef struct Tape {
    Value **slots;
    long long count;
    long long capacity;
} Tape;

Tape *create_tape(void);

void add_value(Tape *t, Value *v);

Value *sum_values(Tape *t, Value *v1, Value *v2);

Value *mul_values(Tape *t, Value *v1, Value *v2);

Value *sub_values(Tape *t, Value *v1, Value *v2);

Value *tanh_value(Tape *t, Value *v);

void tape_backward(Tape *t);

void tape_reset(Tape *t);

void tape_free(Tape *t);

#endif