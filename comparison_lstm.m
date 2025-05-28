%% 全局设置
clc; clear; close all;
data = load('exchanger.dat');
t = data(:,1);
q = data(:,2)';           % 输入：液体流速（行向量）
th = data(:,3)';          % 输出：出口温度（行向量）

%% 数据预处理（统一归一化）
[q_norm, q_ps] = mapminmax(q, 0, 1);
[th_norm, th_ps] = mapminmax(th, 0, 1);

%% 实验1：MLP人工神经网络（使用q(t)和th(t-1)作为输入）
% 数据准备
X_mlp = [q_norm(2:end); th_norm(1:end-1)]';  % 2个输入特征
Y_mlp = th_norm(2:end)';

% 数据划分
N = size(X_mlp,1);
train_idx = 1:round(0.7*N);
test_idx = round(0.7*N)+1:N;

% 转换为数值数组（MLP不需要cell格式）
XTrain_mlp = X_mlp(train_idx,:);
YTrain_mlp = Y_mlp(train_idx,:);
XTest_mlp = X_mlp(test_idx,:);
YTest_mlp = Y_mlp(test_idx,:);

% 定义MLP网络
mlp_layers = [ ...
    featureInputLayer(2)
    fullyConnectedLayer(100)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer];

options_mlp = trainingOptions('adam', ...
    'MaxEpochs', 500, ...
    'InitialLearnRate', 0.001, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

% 训练MLP
net_mlp = trainNetwork(XTrain_mlp, YTrain_mlp, mlp_layers, options_mlp);

% 测试MLP
YPred_mlp_norm = predict(net_mlp, XTest_mlp);
y_pred_mlp = mapminmax('reverse', YPred_mlp_norm, th_ps);
y_true_mlp = mapminmax('reverse', YTest_mlp, th_ps);

% 评估MLP
evaluate_model('MLP', y_true_mlp, y_pred_mlp, t(test_idx+1));

%% 实验2：单变量LSTM-1（仅使用q(t)作为输入）
% 数据准备
X_lstm1 = q_norm(2:end)';  % 仅使用q(t)作为输入
Y_lstm1 = th_norm(2:end)';

% 转换为cell格式
XTrain_lstm1 = {X_lstm1(train_idx)'};
YTrain_lstm1 = {Y_lstm1(train_idx)'};
XTest_lstm1 = {X_lstm1(test_idx)'};
YTest_lstm1 = Y_lstm1(test_idx)';

% 定义LSTM网络
lstm1_layers = [ ...
    sequenceInputLayer(1)
    lstmLayer(50, 'OutputMode', 'sequence')
    fullyConnectedLayer(1)
    regressionLayer];

options_lstm = trainingOptions('adam', ...
    'MaxEpochs', 300, ...
    'InitialLearnRate', 0.005, ...
    'GradientThreshold', 1, ...
    'Verbose', 0);

% 训练LSTM1
net_lstm1 = trainNetwork(XTrain_lstm1, YTrain_lstm1, lstm1_layers, options_lstm);

% 测试LSTM1
YPred_lstm1_norm = predict(net_lstm1, XTest_lstm1);
y_pred_lstm1 = mapminmax('reverse', YPred_lstm1_norm{1}, th_ps);
y_true_lstm1 = mapminmax('reverse', YTest_lstm1, th_ps);

% 评估LSTM1
evaluate_model('LSTM-Q', y_true_lstm1, y_pred_lstm1, t(test_idx+1));

%% 实验3：单变量LSTM-2（仅使用th(t-1)作为输入）
% 数据准备
X_lstm2 = th_norm(1:end-1)';  % 仅使用th(t-1)作为输入
Y_lstm2 = th_norm(2:end)';

% 转换为cell格式
XTrain_lstm2 = {X_lstm2(train_idx)'};
YTrain_lstm2 = {Y_lstm2(train_idx)'};
XTest_lstm2 = {X_lstm2(test_idx)'};
YTest_lstm2 = Y_lstm2(test_idx)';

% 定义LSTM网络（使用相同结构）
lstm2_layers = [ ...
    sequenceInputLayer(1)
    lstmLayer(50, 'OutputMode', 'sequence')
    fullyConnectedLayer(1)
    regressionLayer];

% 训练LSTM2
net_lstm2 = trainNetwork(XTrain_lstm2, YTrain_lstm2, lstm2_layers, options_lstm);

% 测试LSTM2
YPred_lstm2_norm = predict(net_lstm2, XTest_lstm2);
y_pred_lstm2 = mapminmax('reverse', YPred_lstm2_norm{1}, th_ps);
y_true_lstm2 = mapminmax('reverse', YTest_lstm2, th_ps);

% 评估LSTM2
evaluate_model('LSTM-TH', y_true_lstm2, y_pred_lstm2, t(test_idx+1));

%% 通用评估函数
function evaluate_model(model_name, y_true, y_pred, t_test)
    mse_val = mean((y_true - y_pred).^2);
    rmse_val = sqrt(mse_val);
    mae_val = mean(abs(y_true - y_pred));
    r2_val = 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2);
    
    fprintf('\n%s 测试集评估指标:\n', model_name);
    fprintf('MSE: %.4f\n', mse_val);
    fprintf('RMSE: %.4f\n', rmse_val);
    fprintf('MAE: %.4f\n', mae_val);
    fprintf('R²: %.4f\n', r2_val);
    
    figure('Position', [100, 100, 2000, 500]);
    plot(t_test, y_true, 'r-', 'LineWidth', 1.5); hold on;
    plot(t_test, y_pred, 'b--', 'LineWidth', 1.5);
    xlabel('时间');
    ylabel('出口温度 (°C)');
    legend('实际输出', '预测输出');
    title([model_name '预测对比']);
    grid on;
    xlim([t_test(1), t_test(end)]);
end