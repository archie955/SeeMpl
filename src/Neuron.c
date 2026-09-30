#include "Neuron.h"
#include <stdlib.h>

struct Neuron *create_neuron(int n_in) {
    struct Neuron *n = malloc(sizeof(struct Neuron));
    if (n == NULL) {
        return NULL;
    }
    n->w = malloc(n_in * sizeof(double));
    if (n->w == NULL) {
        free(n);
        return NULL;
    }
    for (int i = 0; i < n_in; i++) {
        n->w[i] = ((double)rand()/RAND_MAX - 0.5) * 2.0;
    }
    n->b = ((double)rand()/RAND_MAX - 0.5) * 2.0;

    n->w_leaf = malloc(n_in * sizeof(struct Value *));
    return n;
}

struct Value *neuron_act(struct Neuron *n, struct Value **x, struct Tape *t) {
    struct Value *b_leaf = init(n->b);
    add_value(t, b_leaf);
    n->b_leaf = b_leaf;

    struct Value *acc = b_leaf;
    for (int i = 0; i < n->n_in; i++) {
        struct Value *w_leaf = init(n->w[i]);
        add_value(t, w_leaf);
        n->w_leaf[i] = w_leaf;

        struct Value *prod = mul_values(t, w_leaf, x[i]);
        acc = sum_values(t, acc, prod);
    }

    return tanh_value(t, acc);
}

void neuron_update(struct Neuron *n, double lr) {
    n->b -= lr * n->b_leaf->grad;
    for (int i = 0; i < n->n_in; i++) {
        n->w[i] -= lr * n->w_leaf[i]->grad;
    }
}