#include <math.h>
#include <stdlib.h>
#include "1dValue.h"

void default_backward() {
    return;
}

struct Value *init(double data) {
    struct Value *out = malloc(sizeof(struct Value));
    if (out == NULL) {
        return NULL;
    }
    out->data = data;
    out->grad = 0.0;
    out->backward = &default_backward;
    out->num_prev = 0;

    return out;
}

void add_backward(Value *v) {
    v->prev[0]->grad += v->grad;
    v->prev[1]->grad += v->grad;
}

void mul_backward(Value *v) {
    v->prev[0]->grad += v->prev[1]->data * v->grad;
    v->prev[1]->grad += v->prev[0]->data * v->grad;
}

void tanh_backward(Value *v) {
    v->prev[0]->grad += v->grad * (1.0 - v->data * v->data);
}

Value *add(Value *v1, Value *v2) {
    Value *out = malloc(sizeof(struct Value) + 2*sizeof(struct Value*));
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
    Value *out = malloc(sizeof(struct Value) + 2*sizeof(struct Value*));
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

Value *vtanh(Value *v) {
    Value *out = malloc(sizeof(struct Value) + sizeof(struct Value*));
    if (out == NULL) {
        return NULL;
    }

    out->data = tanh(v->data);
    out->grad = 0.0;
    out->backward = tanh_backward;
    out->num_prev = 2;
    out->prev[0] = v;

    return out;
}