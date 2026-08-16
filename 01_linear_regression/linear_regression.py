# Training data
x_train = [1000, 1500, 2000, 2500]
y_train = [200, 300, 400, 500]


# Prediction
def predict(x, w, b):
    return w * x + b


def compute_cost(x_train, y_train, w, b):
    total_cost = 0
    m = len(x_train)

    for i in range(m):
        predicted_value = predict(x_train[i], w, b)

        error = predicted_value - y_train[i]

        total_cost += error ** 2

    return total_cost / (2 * m)


def compute_gradient(x_train, y_train, w, b):
    dj_dw = 0
    dj_db = 0
    m = len(x_train)

    for i in range(m):
        # prediction
        prediction = predict(x_train[i], w, b)

        # calculate cost
        error = prediction - y_train[i]

        dj_dw += error * x_train[i]
        dj_db += error
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_decent(x_train, y_train, w, b, alpha, iteration):

    for i in range(iteration):
        dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b


# Initial parameters
w = 0
b = 0

# Learning rate
alpha = 0.00000001

# Number of iterations
iterations = 10000


# Train model
w, b = gradient_decent(x_train, y_train, w, b, alpha, iterations)

print("w =", w)
print("b =", b)

# Test prediction
prediction = predict(1000, w, b)

print("Prediction for 1800 sq ft:", prediction)

# Calculate final cost
cost = compute_cost(x_train, y_train, w, b)

print("Final cost:", cost)