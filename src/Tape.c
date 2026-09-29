#include "1dValue.h"
#include "Tape.h"

Tape *create_tape(void) {
    struct Tape out = {0};
    return &out; 
}

void add_value(Tape *t, Value *v) {
    t->slots[t->count] = v;
    t->count += 1;
}

Value *sum_values(Tape *t, Value *v1, Value *v2) {
    Value *out = add(v1, v2);
    add_value(t, out);
    return out;
}

Value *mul_values(Tape *t, Value *v1, Value *v2) {
    Value *out = mul(v1, v2);
    add_value(t, out);
    return out;
}

Value *sub_values(Tape *t, Value *v1, Value *v2) {
    Value *negative = init(-1.0);
    add_value(t, negative);

    Value *intermediate = mul_values(t, v2, negative);

    Value *out = sum_values(t, v1, intermediate);

    return out;
}

Value *tanh_value(Tape *t, Value *v) {
    Value *out = vtanh(v);
    add_value(t, out);
    return out;
}

void tape_backward(Tape *t) {
    for (unsigned long long i = t->count - 1; i >= 0; i--) {
        t->slots[i]->backward(t->slots[i]);
    }
}

void tape_reset(Tape *t) {
    for (unsigned long long i = 0; i < t->count; i++) {
        t->slots[i]->grad = 0.0;
    }
}

void tape_update(Tape *t) {
    for (unsigned long long i = t->count - 1; i >= 0; i--) {
        t->slots[i]->data -= 0.1 * t->slots[i]->grad;
    }
}