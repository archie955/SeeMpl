#include "Layer.h"

Layer *create_layer(int count, int prev_count, Tape *t) {
    Layer *l = malloc(sizeof(struct Layer));
    if (l == NULL) {
        return NULL;
    }

    l->n = malloc(count * sizeof(struct Neuron));
    if (l->n == NULL) {
        free(l);
        return NULL;
    }
    for (int i = 0; i < count; i++) {
        l->n[i] = create_neuron(prev_count);
    }
    l->count = count;
    return l;
}

Value **layer_act(struct Layer *l, Value **x, Tape *t) {
    Value **out = malloc(l->count * sizeof(struct Value));
    if (out == NULL) {
        return NULL;
    }
    for (int i = 0; i < l->count; i++) {
        out[i] = neuron_act(l->n[i], x, t);
    }
    return out;
}