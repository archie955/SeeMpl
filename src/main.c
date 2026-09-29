#include "MLP.h"
#include <stdio.h>

int main() {
    Tape *t = create_tape();
    int *size = {3, 4, 4, 1};
    int count = 4;
    Value *r1 = {init(2.0), init(3.0), init(-1.0)};
    Value *r2 = {init(3.0), init(-1.0), init(0.5)};
    Value *r3 = {init(0.5), init(1.0), init(1.0)};
    Value *r4 = {init(1.0), init(1.0), init(-1.0)};
    for (int i = 0; i < 3; i++) {
        add_value(t, &r1[i]);
        add_value(t, &r2[i]);
        add_value(t, &r3[i]);
        add_value(t, &r4[i]);
    }
    Value **xs = {r1, r2, r3, r4};
    Value *ys = {init(1.0), init(-1.0), init(-1.0), init(1.0)};
    for (int i = 0; i < 4; i++) {
        add_value(t, &ys[i]);
    }
    MLP *mlp = create_mlp(size, count, t);
    Value *loss_val = init(0.0);

    for (int k = 0; k < 5000; k++) {
        loss_val = loss(mlp, xs, ys, mlp->count, t);
        tape_reset(t);
        tape_backward(t);
        tape_update(t);
    }
    printf("loss is %d", loss_val->data);
    return 0;
}