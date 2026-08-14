# Training data
x_train = [1000, 1500, 2000, 2500]
y_train = [200, 300, 400, 500]


# Prediction
def predict(x, w, b):
    return w * x + b

def compute_cost(x_train, y_train, w, b):
    total_cost = 0
    length_of_input = len(x_train)

    for i in range(length_of_input):
        predicted_value = predict(x_train[i], w, b)

        error = predicted_value - y_train[i]

        total_cost += error ** 2

    return total_cost / (2 * length_of_input)

# Model parameters
w = 0.2
b = -10


cost = compute_cost(x_train, y_train, w, b)

print("COST", cost)
