#include "Layer.h"

Layer *create_layer(int count, int prev_count) {
    Layer *l = malloc(sizeof(struct Layer));
    if (l == NULL) {
        return NULL;
    }

    l->n = malloc(count * sizeof(struct Neuron *));
    if (l->n == NULL) {
        free(l);
        return NULL;
    }
    for (int i = 0; i < count; i++) {
        l->n[i] = create_neuron(prev_count);
    }
    l->n_in = count;
    l->n_prev = prev_count;
    return l;
}

Value **layer_act(struct Layer *l, Value* x[l->n_prev], Tape *t) {
    Value **out = malloc(l->n_in * sizeof(struct Value *));
    if (out == NULL) {
        return NULL;
    }
    for (int i = 0; i < l->n_in; i++) {
        out[i] = neuron_act(l->n[i], x, t);
    }
    return out;
}

void layer_free(struct Layer *l) {
    for (int i = 0; i < l->n_in; i++) {
        neuron_free(l->n[i]);
    }
    free(l->n);
    free(l);
}