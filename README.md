# Project SOUL

Contributors: Yejun Jang, Juntae Kim, Inhoo Suh

## Introduction

Project SOUL stands for "Seek Our Ultimate Limits" - it is a multidisciplinary project modelling international relations and economics, where the goal is to simulate a currency war using reinforcement learning to derive new economical & political strategies.

We use Multi-Agent Reinforcement Learning (MARL) to model each country as an agent. Between agents, an affinity network is defined and used to determine the shared reward between agents. The agents may send and accept invitations to increase mutual affinity in each time step.

Currently, we have built a DQN-based RL Agent class (DQNAgent) and tested its ability to cooperate in multi-agent scenarios using PettingZoo. PettingZoo is a multi-agent environment library, akin to OpenAI gym, which lets agents to interact with each other while collecting rewards.

## Progress



