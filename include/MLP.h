#ifndef FILE_MLP_H
#define FILE_MLP_H

#include "Layer.h"

typedef struct MLP {
    int *size;
    Layer **l;
    int count;
} MLP;

MLP *create_mlp(int *size, int count, Tape *t);

Value *mlp_act(struct MLP *mlp, Value **x, Tape *t);

Value *loss(MLP *mlp, Value **xs, Value *ys, int count, Tape *t);

#endif