# MSSP

Explores combinations of primitive feature transformations (log, sqrt, exp, reciprocal, square, etc.) and synergies between them.

Primitive Level – fits an independent LinearRegression on every single transformed feature.

Cross Levels – Combines the best models from the previous level into cross‑nodes (multi‑feature linear regressors) and evaluates them.

At each level the population is filtered while keeping a small slice of weak models for diversity.The process repeats until the desired number of levels is reached or there is no decrease in the loss function.
