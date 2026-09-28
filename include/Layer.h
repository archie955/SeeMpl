#include "Neuron.h"

typedef struct Layer {
    Neuron **n;
    int count;
} Layer;

Layer *create_layer(int count, int prev_count, Tape *t) {
    Layer *l =(Layer*) malloc(sizeof(Layer));
    if (l == NULL) {
        free(l);
        return NULL;
    }

    l->n =(Neuron**) malloc(count * sizeof(Neuron));
    if (l->n == NULL) {
        free(l->n);
        return NULL;
    }
    for (int i = 0; i < count; i++) {
        l->n[i] = create_neuron(prev_count, t);
    }
    l->count = count;
    return l;
}

Value **layer_act(struct Layer *l, double*x, Tape *t) {
    Value **out =(Value**) malloc(l->count * sizeof(Value));
    if (out == NULL) {
        free(out);
        return NULL;
    }
    for (int i = 0; i < l->count; i++) {
        out[i] = neuron_act(l->n[i], x, t);
    }
    return out;
}