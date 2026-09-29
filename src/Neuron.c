#include "Neuron.h"
#include <stdlib.h>

#define RANDMAX 2147483647

Neuron *create_neuron(int count, Tape *t) {
    Neuron *n =(Neuron*) malloc(sizeof(Neuron));
    if (n == NULL) {
        free(n);
        return NULL;
    }
    n->w =(Value**) malloc(count * sizeof(Value));
    if (n->w == NULL) {
        return NULL;
    }
    for (int i = 0; i < count; i++) {
        n->w[i] = init(rand()/RANDMAX);
        add_value(t, n->w[i]);
    }
    n->b = init(rand()/RANDMAX);
    add_value(t, n->b);
    
    n->count = count;
    return n;
}

Value *neuron_act(struct Neuron *n, double *x, Tape *t) {
    Value *sum = n->b;
    int count = n->count;
    for (int i = 0; i < count; i++) {
        Value *intermediate = init(x[i]);
        add_value(t, intermediate);
        sum = sum_values(t, sum, mul_values(t, n->w[i], intermediate));
    }
    return tanh_value(t, sum);
}