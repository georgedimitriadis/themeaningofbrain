

import numpy as np


def looped_regression(X, Y, model, leave_percentage_out, verbose=False):
    s = len(Y)
    leave_amount_out = int(leave_percentage_out * s)

    y_pred = np.empty(0)
    start_sample = 0
    while start_sample + leave_amount_out <= s:
        test_samples = np.arange(start_sample, start_sample + leave_amount_out)
        train_samples = np.delete(np.arange(s), test_samples)
        start_sample += leave_amount_out
        if verbose:
            print(start_sample, start_sample + leave_amount_out)

        x_train = X[train_samples, :]
        y_train = np.array(Y)[train_samples]
        x_test = X[test_samples, :]

        model.fit(x_train, y_train)
        y_pred = np.concatenate((y_pred, model.predict(x_test)))

    error = np.linalg.norm(Y[:len(y_pred)] - y_pred)
    print(error)

    return y_pred


def add_past_onto_X(X, past_steps):

    number_of_neurons = X.shape[0]
    number_of_steps = X.shape[1]

    X_past = np.empty((number_of_neurons * past_steps, number_of_steps))

    for n in np.arange(number_of_neurons):
        for s in np.arange(past_steps):
            past_spike_counts = np.zeros(number_of_steps)
            if s == 0:
                past_spike_counts = X[n]
            else:
                past_spike_counts[s:] = X[n, :-s]

            X_past[n * past_steps + s, :] = past_spike_counts

    return X_past


