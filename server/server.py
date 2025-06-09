# server/server.py
import flwr as fl

# 간단한 전략 정의 (모든 클라이언트 평균)
strategy = fl.server.strategy.FedAvg()

# 서버 시작
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
