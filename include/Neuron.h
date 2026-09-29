#include "Tape.h"
#include <stdlib.h>
#define RANDMAX 2147483647


typedef struct Neuron {
    Value **w;
    Value *b;
    int count;
} Neuron;

Neuron *create_neuron(int count, Tape *t);

Value *neuron_act(struct Neuron *n, double *x, Tape *t);

