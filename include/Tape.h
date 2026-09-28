#include "1dValue.h"

typedef struct Tape {
    Value **slots;
    unsigned long long count;
} Tape;

Tape *create_tape(void) {
    struct Tape out = {0};
    return &out; 
}

void reset_tape(Tape *t) {
    for (unsigned long long i = 0; i < t->count; i++) {
        t->slots[i]->grad = 0.0;
    }
}

void add_value(Tape *t, Value *v) {
    t->slots[t->count] = v;
    t->count += 1;
}

void sum_values(Tape *t, Value *v1, Value *v2) {
    Value *out = add(v1, v2);
    add_value(t, out);
}

void mul_values(Tape *t, Value *v1, Value *v2) {
    Value *out = mul(v1, v2);
    add_value(t, out);
}

void tanh_value(Tape *t, Value *v) {
    Value *out = tanh(v);
    add_value(t, out);
}

void tape_backward(Tape *t) {
    for (unsigned long long i = t->count - 1; i >= 0; i--) {
        t->slots[i]->backward(t->slots[i]);
    }
}