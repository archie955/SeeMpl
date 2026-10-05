import build.lib.cnn as cnn
from micrograd.nn import MLP
import time

def short_micrograd_run():
    model = MLP(3, [4, 4, 1])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]

    ys = [1.0, -1.0, -1.0, 1.0]

    for k in range(5000):
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            p.data -= 0.01 * p.grad

    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    return loss.data

def short_cnn_run():
    n_layers = 4
    size = [3, 4, 4, 1]
    n_train = 4
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    n_steps = 5000
    lr = 0.01
    res = cnn.run(n_layers, size, n_train, xs, ys, n_steps, lr)
    return res["loss"]

def benchmark_micrograd(n_times=5):
    times = []
    losses = []
    for i in range(n_times):
        print(f"Running micrograd benchmark iteration {i+1}/{n_times}\n")
        start = time.time()
        loss = short_micrograd_run()
        end = time.time()
        times.append(end - start)
        losses.append(loss)

    avg_time = sum(times) / len(times)
    avg_loss = sum(losses) / len(losses)
    print(f"Micrograd: Average time: {avg_time:.4f}s, Average loss: {avg_loss:.4f}")

def benchmark_cnn(n_times=5):
    times = []
    losses = []
    for i in range(n_times):
        print(f"Running CNN benchmark iteration {i+1}/{n_times}\n") 
        start = time.time()
        loss = short_cnn_run()
        end = time.time()
        times.append(end - start)
        losses.append(loss)

    avg_time = sum(times) / len(times)
    avg_loss = sum(losses) / len(losses)
    print(f"CNN: Average time: {avg_time:.4f}s, Average loss: {avg_loss:.4f}")

def main(n_times=5):
    print("Running benchmarks...")
    benchmark_micrograd(n_times)
    benchmark_cnn(n_times)
    print("Benchmarks completed.")

if __name__ == "__main__":
    main()