#include "MLP.h"

MLP *create_mlp(int *size, int count, Tape *t) {
    MLP *mlp = malloc(sizeof(struct MLP));
    if (mlp == NULL) {
        return NULL;
    }

    mlp->l = malloc((count - 1) * sizeof(struct Layer *));
    if (mlp->l == NULL) {
        free(mlp);
        return NULL;
    }

    for (int i = 0; i < count - 1; i++) {
        mlp->l[i] = create_layer(size[i+1], size[i], t);
    }
    mlp->size = size;
    mlp->count = count;

    return mlp;
}

Value *mlp_act(struct MLP *mlp, Value **x, Tape *t) {
    for (int i = 0; i < mlp->count; i++) {
        x = layer_act(mlp->l[i], x, t);
    }
    return x;
}

Value *loss(MLP *mlp, Value **xs, Value *ys, int count, Tape *t) {
    Value **ypred = malloc(mlp->l[0]->count * count * sizeof(struct Value));
    if (ypred == NULL) {
        return init(0.0);
    }
    
    for (int i = 0; i < count; i++) {
        ypred[i] = mlp_act(mlp, xs[i], t);
    }
    Value *vl = init(0.0);
    add_value(t, vl);
    for (int j = 0; j < count; j++) {
        for (int k = 0; k < mlp->l[j]->count; k++) {
            Value *diff = sub_values(t, &ypred[j][k], &ys[j]);
            Value *loss_part = mul_values(t, diff, diff);
            vl = sum_values(t, vl, loss_part);
        }
    }
    return vl;
}