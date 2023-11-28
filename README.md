# Project SOUL

Contributors: Yejun Jang, Juntae Kim, Inhoo Suh

## Introduction

Project SOUL stands for "Seek Our Ultimate Limits" - it is a multidisciplinary project modelling international relations and economics, where the goal is to simulate a currency war using reinforcement learning to derive new economical & political strategies.

We use Multi-Agent Reinforcement Learning (MARL) to model each country as an agent. Between agents, an affinity network is defined and used to determine the shared reward between agents. The agents may send and accept invitations to increase mutual affinity in each time step.

Currently, we have built a DQN-based RL Agent class (DQNAgent) and tested its ability to cooperate in multi-agent scenarios using PettingZoo. PettingZoo is a multi-agent environment library, akin to OpenAI gym, which lets agents to interact with each other while collecting rewards.

## Progress

Below are the descriptions for important directories:

**.Source/CurrencyPricePredictionLSTM:** We examined the predictive capacity of Long Short-Term Memory (LSTM) models in forecasting currency prices. Based on our analysis, predictions using LSTM models and OHLCV (Open, High, Low, Close, and Volume) data did not provide adequate predictive accuracy. As a result, we have shifted our focus to multi-agent reinforcement learning and adopting a simulation-based approach.

**.Source/PreprocessingInternationalRelationsData:** In our simulation, agents can adjust the amount of rewards to share by adjusting their affinity. However, the initial affinity should be able to reflect the current diplomacies. We used Defense Cooperation Agreement Dataset by Brandon J. Kinne, and a web-parsed version of International Organization Participation data from cia.gov to create realistic affinity graphs between different countries.

**.Source/MultiAgentDeepQLearning:** We use Deep Q-Network (DQN) architectures and their variants to model agents. Reasons for choosing DQN includes ease of analysis, decent performance and popularity. Most RL setups assume stationarity of the transition dynamics, meaning they assume the laws underlying the environment do not change. However, introducing multiple agents requires that one agent can adapt to other agent's strategic change, which introduces further challenge.

**.Source/PrototypeVer1:** A working prototype of PPO agents learning in an economical environment. Each country(=agent) can set its own interest rate at each time step. The right policy will maximize the share of GDP at the end of the 