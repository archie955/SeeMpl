#include "1dValue.h"
#include "Tape.h"

Tape *create_tape(void) {
    struct Tape *t = malloc(sizeof(struct Tape));
    if (t == NULL) {
        return NULL;
    }

    t->capacity = 64;
    t->count = 0;
    t->slots = malloc(t->capacity * sizeof(struct Value *));
    if (t->slots == NULL) {
        free(t);
        return NULL;
    }
    return t; 
}

void add_value(Tape *t, Value *v) { // this is what python does under the hood
    if (t->count == t->capacity) {
        long long new_cap = t->capacity * 2;
        Value **new_slots = realloc(t->slots, new_cap * sizeof(Value *));
        if (new_slots == NULL) {
            return NULL;
        }
        t->slots = new_slots;
        t->capacity = new_cap;
    }
    t->slots[t->count++] = v;
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
    for (long long i = t->count - 1; i >= 0; i--) {
        t->slots[i]->backward(t->slots[i]);
    }
}

void tape_reset(Tape *t) {
    for (long long i = 0; i < t->count; i++) {
        t->slots[i]->grad = 0.0;
    }
}

void tape_update(Tape *t) {
    for (long long i = t->count - 1; i >= 0; i--) {
        t->slots[i]->data -= 0.1 * t->slots[i]->grad;
    }
}