#include "MLP.h"
#include <stdio.h>

void mlp_train_step(MLP *mlp, double *x_raw, int n_in, double y_raw, double lr) {
    Tape *t = mlp->t;
    tape_reset(t);

    Value *x[n_in];
    for (int i = 0; i < n_in; i++) {
        x[i] = init(x_raw[i]);
        add_value(t, x[i]);
    }

    Value **pred = mlp_act(mlp, x);

    Value *y = init(y_raw);
    add_value(t, y);
    Value *diff = sub_values(t, pred[0], y);
    Value *loss = mul_values(t, diff, diff);
    
    loss->grad = 1.0;
    tape_backward(t);

    mlp_update(mlp, lr);
}

int main() {
    Tape *t = create_tape();
    int size[4] = {3, 4, 4, 1};
    int count = 4;
    Value *r1[3] = {init(2.0), init(3.0), init(-1.0)};
    Value *r2[3] = {init(3.0), init(-1.0), init(0.5)};
    Value *r3[3] = {init(0.5), init(1.0), init(1.0)};
    Value *r4[3] = {init(1.0), init(1.0), init(-1.0)};
    for (int i = 0; i < 3; i++) {
        add_value(t, &r1[i]);
        add_value(t, &r2[i]);
        add_value(t, &r3[i]);
        add_value(t, &r4[i]);
    }
    Value* xs[4] = {r1, r2, r3, r4};
    Value *ys[4] = {init(1.0), init(-1.0), init(-1.0), init(1.0)};
    for (int i = 0; i < 4; i++) {
        add_value(t, &ys[i]);
    }
    MLP *mlp = create_mlp(size, count);
    Value *loss_val = init(0.0);

    for (int k = 0; k < 5000; k++) {
        loss_val = loss(mlp, xs, ys, mlp->count);
        tape_reset(t);
        tape_backward(t);
        tape_update(t);
    }
    printf("loss is %f", loss_val->data);
    return 0;
}