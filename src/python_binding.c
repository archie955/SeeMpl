#if __has_include(<Python.h>)
#include <Python.h>
#elif __has_include(<python3.14/Python.h>)
#include <python3.14/Python.h>
#elif __has_include(<python3.13/Python.h>)
#include <python3.13/Python.h>
#elif __has_include(<python3.12/Python.h>)
#include <python3.12/Python.h>
#elif __has_include(<python3.11/Python.h>)
#include <python3.11/Python.h>
#elif __has_include(<python3.10/Python.h>)
#include <python3.10/Python.h>
#elif __has_include(<python3.9/Python.h>)
#include <python3.9/Python.h>
#elif __has_include(<python3.8/Python.h>)
#include <python3.8/Python.h>
#else
#error "Python development headers not found; install them and configure the compiler include path"
#endif
#include "MLP.h"
#include <stdio.h>

PyObject* create_output(MLP *mlp, double loss) {
    PyObject *dict = PyDict_New();
    if (dict == NULL) {
        return NULL;
    }

    PyObject *loss_obj = PyFloat_FromDouble(loss);
    if (loss_obj == NULL) {
        Py_DECREF(dict);
        return NULL;
    }

    if (PyDict_SetItemString(dict, "loss", loss_obj) < 0) {
        Py_DECREF(loss_obj);
        Py_DECREF(dict);
        return NULL;
    }

    Py_DECREF(loss_obj);

    for (int i = 0; i < mlp->count - 1; i++) {
        Layer *l = mlp->l[i];

        PyObject *layer = PyList_New(l->n_in);
        if (layer == NULL) {
            Py_DECREF(dict);
            return NULL;
        }

        for (int j = 0; j < l->n_in; j++) {
            Neuron *n = l->n[j];

            PyObject *weights = PyList_New(n->n_in);
            if (weights == NULL) {
                Py_DECREF(layer);
                Py_DECREF(dict);
                return NULL;
            }

            for (int k = 0; k < n->n_in; k++) {
                PyObject *weight = PyFloat_FromDouble(n->w[k]);
                if (weight == NULL) {
                    Py_DECREF(weights);
                    Py_DECREF(layer);
                    Py_DECREF(dict);
                    return NULL;
                }

                PyList_SetItem(weights, k, weight);
            }

            PyObject *neuron = PyTuple_New(2);
            if (neuron == NULL) {
                Py_DECREF(weights);
                Py_DECREF(layer);
                Py_DECREF(dict);
                return NULL;
            }

            PyTuple_SetItem(neuron, 0, weights);
            PyTuple_SetItem(
                neuron,
                1,
                PyFloat_FromDouble(n->b)
            );

            PyList_SetItem(layer, j, neuron);
        }

        char key[32];
        snprintf(key, sizeof(key), "layer_%d", i + 1);

        if (PyDict_SetItemString(dict, key, layer) < 0) {
            Py_DECREF(layer);
            Py_DECREF(dict);
            return NULL;
        }

        Py_DECREF(layer);
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

PyObject* RunMethod(PyObject *self, PyObject *args) {
    int n_layers;
    PyObject *size_list;
    int n_train;
    PyObject *xs_list;
    PyObject *ys_list;
    int n_steps;
    double lr;

    if (!PyArg_ParseTuple(args, "iO!iO!O!id", &n_layers, &PyList_Type, &size_list, &n_train, &PyList_Type, &xs_list, &PyList_Type, &ys_list, &n_steps, &lr)) {
        return NULL;
    }

    int size[n_layers];
    for (int i = 0; i < n_layers; i++) {
        PyObject *item = PyList_GetItem(size_list, i);
        if (item == NULL) {
            Py_DECREF(item);
            return NULL;
        }
        size[i] = (int)PyLong_AsLong(item);
    }

    double xs[n_train][size[0]];
    for (int i = 0; i < n_train; i++) {
        PyObject *row = PyList_GetItem(xs_list, i);
        if (row == NULL) {
            Py_DECREF(row);
            return NULL;
        }
        for (int j = 0; j < size[0]; j++) {
            PyObject *item = PyList_GetItem(row, j);
            if (item == NULL) {
                Py_DECREF(item);
                return NULL;
            }
            xs[i][j] = PyFloat_AsDouble(item);
        }
    }

    double ys[n_train];
    for (int i = 0; i < n_train; i++) {
        PyObject *item = PyList_GetItem(ys_list, i);
        if (item == NULL) {
            Py_DECREF(item);
            return NULL;
        }
        ys[i] = PyFloat_AsDouble(item);
    }

    return run(n_layers, size, n_train, xs, ys, n_steps, lr);
}

static PyMethodDef train[] = {
    {"run", RunMethod, METH_VARARGS, "Train the neural network"},
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