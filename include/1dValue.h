

typedef struct Value {
    double data;
    double grad;
    bool requires_grad;
    void (*backward)(struct Value *self);
    int num_prev;
    struct Value *prev[];
} Value;

