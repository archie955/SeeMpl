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
        mlp->l[i] = create_layer(size[i+1], size[i]);
    }
    mlp->size = size;
    mlp->count = count;

    return mlp;
}

Value** mlp_act(struct MLP *mlp, Value* x[mlp->count]) {
    for (int i = 0; i < mlp->count - 1; i++) { // count is number of layers, but this includes first, so n-1 transformations between layers
        x = layer_act(mlp->l[i], x, mlp->t);
    }
    return x;
}

void mlp_update(MLP *mlp, double lr) {
    for (int i = 0; i < mlp->count - 1; i++) {
        Layer *l = mlp->l[i];
        for (int j = 0; j < l->n_in; j++) {
            neuron_update(l->n[j], lr);
        }
    }
}

double mlp_train_step(MLP *mlp, int n_in, int n_train, double x_raw[n_train][n_in], double y_raw[n_train], double lr) {
    Tape *t = mlp->t;
    tape_reset(t);

    Value* x[n_train][n_in];
    Value *y;
    Value *diff;
    Value *loss_part;
    Value *loss = init(0.0);
    add_value(t, loss); // NOTETOSELF: had memory leak previously by not adding this
    for (int j = 0; j < n_train; j++) {
        for (int i = 0; i < n_in; i++) {
            x[j][i] = init(x_raw[j][i]);
            add_value(t, x[j][i]);
        }

        Value **pred = mlp_act(mlp, x[j]);

        y = init(y_raw[j]);
        add_value(t, y);
        diff = sub_values(t, pred[0], y);
        loss_part = mul_values(t, diff, diff);
        loss = sum_values(t, loss, loss_part);
    }
    
    loss->grad = 1.0;
    tape_backward(t);

    mlp_update(mlp, lr);
    return loss->data;
}