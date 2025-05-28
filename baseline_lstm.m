clc; clear; close all;

%% 1. 读取数据
data = load('exchanger.dat');
t = data(:,1);
q = data(:,2)';           % 输入：液体流速（行向量）
th = data(:,3)';          % 输出：出口温度（行向量）

%% 2. 数据归一化
[q_norm, q_ps] = mapminmax(q, 0, 1);
[th_norm, th_ps] = mapminmax(th, 0, 1);

%% 3. 构建多变量输入：q(t), th(t-1)，输出为 th(t)
q_seq = q_norm(2:end);           % 从 t=2 开始
th_prev_seq = th_norm(1:end-1);  % th(t-1)
X = [q_seq; th_prev_seq];        % 2×(N-1)
Y = th_norm(2:end);              % th(t)

% 按列序列化，转置为 (sequenceLength x numFeatures)
X = X';    % (N-1)×2
Y = Y';    % (N-1)×1

%% 4. 数据划分
N = size(X,1);
train_idx = 1:round(0.7*N);
test_idx = round(0.7*N)+1:N;

XTrain = {X(train_idx,:)'};  % cell 包含 size: 2×T
YTrain = {Y(train_idx,:)'};  % cell 包含 size: 1×T

XTest = {X(test_idx,:)'};
YTest = Y(test_idx,:)';      % 真实输出（归一化）

t_test = t(test_idx+1);      % +1 是因为我们从 t=2 开始构造

%% 5. 定义 LSTM 网络
inputSize = 2;
numHiddenUnits = 100;
outputSize = 1;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
    fullyConnectedLayer(outputSize)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 500, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 100, ...
    'LearnRateDropFactor', 0.5, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

%% 6. 训练网络
net = trainNetwork(XTrain, YTrain, layers, options);

%% 7. 测试集预测
YPred_norm = predict(net, XTest, 'MiniBatchSize', 1);
y_pred = mapminmax('reverse', YPred_norm{1}, th_ps);
y_true = mapminmax('reverse', YTest, th_ps);

%% 8. 绘图
figure('Position', [100, 100, 2000, 500]);
plot(t_test, y_true, 'r-', 'LineWidth', 1.5); hold on;
plot(t_test, y_pred, 'b--', 'LineWidth', 1.5);
xlabel('时间');
ylabel('出口温度 (°C)');
legend('实际输出', '预测输出');
title('多变量 LSTM：测试集实际 vs 预测');
grid on;
xlim([t_test(1), t_test(end)]);

%% 9. 评估指标
mse_val = mean((y_true - y_pred).^2);
rmse_val = sqrt(mse_val);
mae_val = mean(abs(y_true - y_pred));
r2_val = 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);

fprintf('\n多变量 LSTM 测试集评估指标:\n');
fprintf('MSE: %.4f\n', mse_val);
fprintf('RMSE: %.4f\n', rmse_val);
fprintf('MAE: %.4f\n', mae_val);
fprintf('R^2: %.4f\n', r2_val);
