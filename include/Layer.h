#ifndef FILE_LAYER_H
#define FILE_LAYER_H

#include "Neuron.h"

typedef struct Layer {
    Neuron **n;
    int count;
} Layer;

Layer *create_layer(int count, int prev_count, Tape *t);

Value **layer_act(struct Layer *l, Value **x, Tape *t);

#endif