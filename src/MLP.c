#include "MLP.h"

MLP *create_mlp(int *size, int count) {
    MLP *mlp = malloc(sizeof(struct MLP));
    if (mlp == NULL) {
        return NULL;
    }

    mlp->l = malloc((count - 1) * sizeof(struct Layer *));
    if (mlp->l == NULL) {
        free(mlp);
        return NULL;
    }

    mlp->t = create_tape();

    for (int i = 0; i < count - 1; i++) {
        mlp->l[i] = create_layer(size[i+1], size[i], mlp->t);
    }
    mlp->size = size;
    mlp->count = count;

    return mlp;
}

Value *mlp_act(struct MLP *mlp, Value **x) {
    for (int i = 0; i < mlp->count; i++) {
        x = layer_act(mlp->l[i], x, mlp->t);
    }
    return x;
}

Value *loss(MLP *mlp, Value **xs, Value *ys, int count) {
    Value **ypred = malloc(mlp->l[0]->count * count * sizeof(struct Value));
    if (ypred == NULL) {
        return init(0.0);
    }
    
    for (int i = 0; i < count; i++) {
        ypred[i] = mlp_act(mlp, xs[i]);
    }
    Value *vl = init(0.0);
    add_value(mlp->t, vl);
    for (int j = 0; j < count; j++) {
        for (int k = 0; k < mlp->l[j]->count; k++) {
            Value *diff = sub_values(mlp->t, &ypred[j][k], &ys[j]);
            Value *loss_part = mul_values(mlp->t, diff, diff);
            vl = sum_values(mlp->t, vl, loss_part);
        }
    }
    return vl;
}

Value *mlp_train_step(MLP *mlp, double **x_raw, int n_in, double *y_raw, int n_train, double lr) {
    Tape *t = mlp->t;
    tape_reset(t);

    Value **x[n_train][n_in];
    Value *y;
    Value *diff;
    Value *loss_part;
    Value *loss = init(0.0);
    for (int j = 0; j < n_train; j++) {
        for (int i = 0; i < n_in; i++) {
            x[j][i] = init(x_raw[j][i]);
            add_value(t, x[j][i]);
        }

        Value **pred = mlp_act(mlp, x);

        y = init(y_raw[j]);
        add_value(t, y);
        diff = sub_values(t, pred[j], y);
        loss_part = mul_values(t, diff, diff);
        loss = sum_values(t, loss, loss_part);
    }
    
    loss->grad = 1.0;
    tape_backward(t);

    tape_update(t, lr);
    return loss;
}