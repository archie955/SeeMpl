#include <math.h>
#include <stdlib.h>

typedef struct Value {
    double data;
    double grad;
    void (*backward)(struct Value *self);
    int num_prev;
    struct Value *prev[];
} Value;

static void default_backward(struct Value *self) {
    return;
}

static struct Value *init(double data) {
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

static double tanh(double x) {
    double e = exp(2*x);
    double t = (e-1) / (e+1);
    return t;
}

static void tanh_backward(Value *v) {
    double x = v->data;
    double t = tanh(x);
    v->prev[0]->grad += v->grad * (1.0 - t*t);
}

static Value *add(Value *v1, Value *v2) {
    Value *out =(Value*) malloc(sizeof(Value));

    out->data = v1->data + v2->data;
    out->grad = 0.0;
    out->backward = add_backward;
    out->num_prev = 2;
    out->prev[0] = v1;
    out->prev[1] = v2;

    return out;
}

static Value *mul(Value *v1, Value *v2) {
    Value *out =(Value*) malloc(sizeof(Value));

    out->data = v1->data * v2->data;
    out->grad = 0.0;
    out->backward = mul_backward;
    out->num_prev = 2;
    out->prev[0] = v1;
    out->prev[1] = v2;

    return out;
}

static Value *tanh(Value *v) {
    Value *out =(Value*) malloc(sizeof(Value));

    out->data = tanh(v->data);
    out->grad = 0.0;
    out->backward = tanh_backward;
    out->num_prev = 2;
    out->prev[0] = v;

    return out;
}