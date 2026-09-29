#include "Neuron.h"
#include <stdlib.h>

Neuron *create_neuron(int count, Tape *t) {
    Neuron *n = malloc(sizeof(struct Neuron));
    if (n == NULL) {
        return NULL;
    }
    n->w = malloc(count * sizeof(struct Value));
    if (n->w == NULL) {
        free(n);
        return NULL;
    }
    for (int i = 0; i < count; i++) {
        n->w[i] = init((double)rand()/RANDMAX);
        add_value(t, n->w[i]);
    }
    n->b = init((double)rand()/RANDMAX);
    add_value(t, n->b);
    
    n->count = count;
    return n;
}

Value *neuron_act(struct Neuron *n, Value **x, Tape *t) {
    Value *sum = n->b;
    int count = n->count;
    for (int i = 0; i < count; i++) {
        sum = sum_values(t, sum, mul_values(t, n->w[i], x[i]));
    }
    return tanh_value(t, sum);
}