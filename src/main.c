#include "MLP.h"
#include <stdio.h>

int main() {
    Tape *t = create_tape();
    int size[4] = {3, 4, 4, 1};
    int count = 4;
    double r1[3] = {2.0, 3.0, -1.0};
    double r2[3] = {3.0, -1.0, 0.5};
    double r3[3] = {0.5, 1.0, 1.0};
    double r4[3] = {1.0, 1.0, -1.0};

    double* xs[4] = {r1, r2, r3, r4};
    double ys[4] = {1.0, -1.0, -1.0, 1.0};
    for (int i = 0; i < 4; i++) {
        add_value(t, &ys[i]);
    }
    MLP *mlp = create_mlp(size, count);
    Value *loss;
    
    for (int i = 0; i < 5000; i++) {
        loss = mlp_train_step(mlp, xs, 3, ys, 4, 0.01);
    }

    printf("loss is %f", loss->data);
    return 0;
}