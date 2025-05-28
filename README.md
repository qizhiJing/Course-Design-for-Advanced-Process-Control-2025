# Course-Design-for-Advanced-Process-Control-2025
# Heat Exchanger System Identification Using LSTM

## 项目简介

本项目基于工业换热器的实验数据，采用长短时记忆网络（LSTM）实现系统辨识建模。通过输入液体流速和前一时刻出口温度，预测当前时刻的出口温度，实现对非线性动态系统的有效建模。

该项目属于高级过程控制课程的大作业，旨在利用深度学习技术解决工业过程控制中的建模难题，提升系统预测精度。

## 数据集

使用的数据集为 [exchanger.dat](https://homes.esat.kuleuven.be/~smc/daisy/daisydata.html)，来源于 Politecnico di Milano 的 Sergio Bittanti 教授提供的液体饱和蒸汽换热器实验数据。  
数据说明详见论文：

- S. Bittanti and L. Piroddi, "Nonlinear identification and control of a heat exchanger: a neural network approach", Journal of the Franklin Institute, 1996.

数据包含：

- 输入：液体流速 (q)  
- 输出：出口液体温度 (th)  
- 采样时间：1秒  
- 样本数：4000

## 运行环境

- MATLAB R202x 或支持深度学习工具箱的版本
- 依赖函数：mapminmax，trainNetwork 等

## 文件说明

- `exchanger.dat` ：数据文件  
- `baseline_lstm` ：模型代码实现
- `comparison_lstm` ：对比方法代码实现
- `README.md` ：项目说明

