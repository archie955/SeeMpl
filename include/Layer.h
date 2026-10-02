#ifndef FILE_LAYER_H
#define FILE_LAYER_H

#include "Neuron.h"

typedef struct Layer {
    Neuron **n;
    int n_in;
    int n_prev;
} Layer;

Layer *create_layer(int count, int prev_count);

Value **layer_act(struct Layer *l, Value* x[l->n_prev], Tape *t);

void layer_free(struct Layer *l);

#endif