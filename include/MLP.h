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

Value **mlp_act(MLP *mlp, Value* x[mlp->count]);

void mlp_update(MLP *mlp, double lr);

Value *mlp_train_step(MLP *mlp, int n_in, int n_train, double x_raw[n_train][n_in], double y_raw[n_train], double lr);

#endif