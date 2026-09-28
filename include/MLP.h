#include "Layer.h"

typedef struct MLP {
    int *size;
    Layer **l;
    int count;
} MLP;

MLP *create_mlp(int *size, int count, Tape *t) {
    MLP *mlp =(MLP*) malloc(sizeof(MLP));
    if (mlp == NULL) {
        free(mlp);
        return NULL;
    }

    mlp->l =(Layer**) malloc((count - 1) * sizeof(Layer *));
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

Value *mlp_act(struct MLP *mlp, Value *x, Tape *t) {
    for (int i = 0; i < mlp->count; i++) {
        x = layer_act(mlp->l[i], x, t);
    }
    return x;
}

Value *loss(MLP *mlp, Value **xs, Value *ys, int count, Tape *t) {
    Value **ypred =(Value**) malloc(mlp->l[0]->count * count * sizeof(Value));
    if (ypred == NULL) {
        free(ypred);
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