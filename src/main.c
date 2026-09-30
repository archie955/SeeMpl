#include "MLP.h"
#include <stdio.h>

int main(void) {
    int size[4] = {3, 4, 4, 1};
    int count = 4;

    double xs[4][3] = {
        {2.0, 3.0, -1.0}, 
        {3.0, -1.0, 0.5},
        {3.0, -1.0, 0.5},
        {3.0, -1.0, 0.5}
    };
    double ys[4] = {1.0, -1.0, -1.0, 1.0};

    MLP *mlp = create_mlp(size, count);
    Value *loss;
    
    for (int i = 0; i < 5000; i++) {
        loss = mlp_train_step(mlp, 3, 4, xs, ys, 0.01);
    }

    printf("loss is %f", loss->data);
    return 0;
}