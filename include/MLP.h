#ifndef FILE_MLP_H
#define FILE_MLP_H

#include "Layer.h"

typedef struct MLP {
    int *size;
    Layer **l;
    int count;
    Tape *t;
} MLP;

MLP *create_mlp(int *size, int count);

Value *mlp_act(MLP *mlp, Value **x);

Value *loss(MLP *mlp, Value **xs, Value *ys, int count);

void mlp_train_step(MLP *mlp, double *x_raw, int n_in, double y_raw, double lr);

#endif