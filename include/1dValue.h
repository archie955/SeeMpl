#ifndef FILE_1DVALUE_H
#define FILE_1DVALUE_H

#include <math.h>
#include <stdlib.h>

typedef struct Value {
    double data;
    double grad;
    void (*backward)(struct Value *self);
    int num_prev;
    struct Value *prev[];
} Value;

static void default_backward(struct Value *self);

struct Value *init(double data);

static void add_backward(Value *v);

static void mul_backward(Value *v);

static double dtanh(double x);

static void tanh_backward(Value *v);

Value *add(Value *v1, Value *v2);

Value *mul(Value *v1, Value *v2);

Value *vtanh(Value *v);

#endif