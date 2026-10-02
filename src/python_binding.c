#include <Python.h>
#include "MLP.h"
#include <stdio.h>

PyObject* create_output(MLP *mlp, double loss) {
        PyObject *dict = PyDict_New();
        PyObject *loss_s = Py_BuildValue("s#", "loss", 4);
        PyDict_SetItem(dict, loss_s, PyFloat_FromDouble(loss));
        for (int i = 0; i < mlp->count - 1; i++) {
            Layer *l = mlp->l[i];
            PyObject *layer = PyList_New(l->n_in);
            for (int j = 0; j < l->n_in; j++) {
                Neuron *n = l->n[j];
                PyObject *neuron = PyTuple_New(2);
                PyObject *weights = PyList_New(n->n_in);
                for (int k = 0; k < n->n_in; k++) {
                    PyList_Append(weights, PyFloat_FromDouble(n->w[k]));
                }
                PyTuple_SetItem(neuron, PyList_Size(weights), weights);
                PyTuple_SetItem(neuron, 24, PyFloat_FromDouble(n->b));
                PyList_Append(layer, neuron);
            }
            char s[32];
            snprintf(s, sizeof(s), "layer_%d", i+1);
            PyObject *str = Py_BuildValue("s#", s, 32);
            PyDict_SetItem(dict, str, layer);
        }
        
        return dict;
}

PyObject* run(int n_layers, int size[n_layers], int n_train, double xs[n_train][size[0]], double ys[n_train], int n_steps, double lr) {
    MLP *mlp = create_mlp(size, n_layers);
    double loss;

    for (int i = 0; i < n_steps; i++) {
        loss = mlp_train_step(mlp, size[0], n_train, xs, ys, lr);
    }


    PyObject *dict = create_output(mlp, loss);
    mlp_free(mlp);
    return dict;
}

static PyMethodDef train[] = {
    {"run", run, METH_VARARGS, "Train the neural network"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cnn = {
    PyModuleDef_HEAD_INIT,
    "cnn",
    "Create and train Neural Network in C",
    -1,
    train
};

PyMODINIT_FUNC PyInit_cnn() {
    return PyModule_Create(&cnn);
}