#include "1dValue.h"

typedef struct Neuron {
    Value **w;
    Value *b;
} Neuron;

int get_count(struct Value **x) {
    return sizeof(x) / sizeof(Value);
} 

double act(struct Neuron *n, struct Value **x) {
    double sum = n->b->data;
    int count = get_count(x);
    for (int i = 0; i < count; i++) {
        sum += n->w[i]->data * x[i]->data;
    }
    return dtanh(sum);
}

