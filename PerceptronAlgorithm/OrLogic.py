import numpy as np
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 1, 1, 0])
initial_weights = np.array([0.3, -0.1])
bias = 0.2
learning_rate = 0.1
def step_function(x):
    return 1 if x >= 0 else 0

# 에폭의 최대값 설정
max_epochs = 10
for epoch in range(1, max_epochs + 1):
    total_error = 0

    # 각 입력에 대한 계산과 최종가중치 업데이트
    for i in range(len(inputs)):
        input_data = inputs[i]
        target = targets[i]

        # 실제 출력 계산
        actual_output = np.dot(input_data, initial_weights) - bias
        actual_output = round(actual_output, 1)
        prediction = step_function(actual_output)

        # 오차 계산
        error = target - prediction
        total_error += abs(error)  # 오차의 절댓값을 누적

        # 최종가중치 업데이트
        initial_weights += learning_rate * error * input_data

        # 에폭별 입력값에 따른 결과 출력
        print(f"Epoch {epoch}, Input: {input_data}, Target: {target}, Actual Output: {actual_output}, Prediction: {prediction}, Error: {error}")
        print(f"  Updated Weights: {initial_weights}\n") 

    # 모든 입력에 대한 누적된 오차가 0이면 종료
    if total_error == 0:
        print(f"Converged at Epoch {epoch}")
        break
# 진행하면서 에폭이 최대 에폭 설정값과 동일하며 오차가 계속하여 존재할 경우 
    if epoch == max_epochs and total_error != 0:
            print("Maximum epochs reached. Did not converge to zero error.")
