#ifndef FILE_NEURON_H
#define FILE_NEURON_H

#include "Tape.h"
#include <stdlib.h>


typedef struct Neuron {
    int n_in;
    double *w;
    double b;
    Value **w_leaf;
    Value *b_leaf;
} Neuron;

Neuron *create_neuron(int n_in);

Value *neuron_act(struct Neuron *n, struct Value* x[n->n_in], struct Tape *t);

void neuron_update(struct Neuron *n, double lr);

void neuron_free(struct Neuron *n);

#endif