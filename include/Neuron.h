#include "1dValue.h"
#include <stdlib.h>
#define RANDMAX 2147483647


typedef struct Neuron {
    Value **w;
    Value *b;
    int count;
} Neuron;

Neuron *create_neuron(int count) {
    Value **w = {};
    for (int i = 0; i < count; i++) {
        w[i] = init(rand()/RANDMAX);
    }
    Value *b = init(rand()/RANDMAX);
    Neuron n = {w, b, count};
    return &n;
}

double act(struct Neuron *n, struct Value **x) {
    double sum = n->b->data;
    int count = n->count;
    for (int i = 0; i < count; i++) {
        sum += n->w[i]->data * x[i]->data;
    }
    return dtanh(sum);
}

