#include <math.h>
#include <stdlib.h>
#include "1dValue.h"

static void default_backward(struct Value *self) {
    return;
}

struct Value *init(double data) {
    struct Value out = {data, 0.0, &default_backward, 0};
    return &out;
}

static void add_backward(Value *v) {
    v->prev[0]->grad += v->grad;
    v->prev[1]->grad += v->grad;
}

static void mul_backward(Value *v) {
    v->prev[0]->grad += v->prev[1]->data * v->grad;
    v->prev[1]->grad += v->prev[0]->data * v->grad;
}

static double dtanh(double x) {
    double e = exp(2*x);
    double t = (e-1) / (e+1);
    return t;
}

static void tanh_backward(Value *v) {
    double x = v->data;
    double t = dtanh(x);
    v->prev[0]->grad += v->grad * (1.0 - t*t);
}

Value *add(Value *v1, Value *v2) {
    Value *out =(Value*) malloc(sizeof(Value));
    if (out == NULL) {
        return NULL;
    }

    out->data = v1->data + v2->data;
    out->grad = 0.0;
    out->backward = add_backward;
    out->num_prev = 2;
    out->prev[0] = v1;
    out->prev[1] = v2;

    return out;
}

Value *mul(Value *v1, Value *v2) {
    Value *out =(Value*) malloc(sizeof(Value));
    if (out == NULL) {
        return NULL;
    }

    out->data = v1->data * v2->data;
    out->grad = 0.0;
    out->backward = mul_backward;
    out->num_prev = 2;
    out->prev[0] = v1;
    out->prev[1] = v2;

    return out;
}

Value *tanh(Value *v) {
    Value *out =(Value*) malloc(sizeof(Value));
    if (out == NULL) {
        return NULL;
    }

    out->data = dtanh(v->data);
    out->grad = 0.0;
    out->backward = tanh_backward;
    out->num_prev = 2;
    out->prev[0] = v;

    return out;
}