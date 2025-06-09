# client1/client.py
import flwr as fl
import numpy as np

# Dummy 데이터 (연합학습 테스트용)
x_train = np.array([[1.0], [2.0], [3.0]])
y_train = np.array([1.0, 2.0, 3.0])

class DummyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [np.array([0.0])]

    def fit(self, parameters, config):
        return parameters, len(x_train), {}

    def evaluate(self, parameters, config):
        return 0.0, len(x_train), {}

fl.client.start_numpy_client(server_address="server:8080", client=DummyClient())
