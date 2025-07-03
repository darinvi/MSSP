import time
import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations
import os
from multiprocessing import Process, Queue

METRIC_FNS = {
    'mae': lambda y, y_pred: np.mean(np.abs(y - y_pred)),
    'mape': lambda y, y_pred: np.mean(np.abs((y - y_pred) / y)) * 100,
}

class Node:
    def __init__(self):
        self.model = LinearRegression()
    
    def evaluate(self, X, y, metric):
        y_pred = self.predict(X)
        return METRIC_FNS[metric](y, y_pred)
    
    def __call__(self, X):
        return self.predict(X)

class PrimitiveNode(Node):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask

    def fit(self, X, y):
        x = X[:, np.argmax(self.mask)].reshape(-1, 1)
        self.model.fit(x, y)
    
    def predict(self, X):
        x = X[:, np.argmax(self.mask)].reshape(-1, 1)
        return self.model.predict(x).flatten()

class CrossNode(Node):
    def __init__(self, children):
        super().__init__()
        self.children = children

    def fit(self, X, y):
        x = self.children[0].predict(X).reshape(-1, 1)
        for c in self.children[1:]:
            x = np.hstack((x, c.predict(X).reshape(-1, 1)))
        self.model.fit(x, y)
    
    def predict(self, X):
        x1 = self.children[0].predict(X).reshape(-1, 1)
        for c in self.children[1:]:
            x1 = np.hstack((x1, c.predict(X).reshape(-1, 1)))
        return self.model.predict(x1).flatten()

class MSSP:
    def __init__(
            self, 
            X,
            y,
            metric='mae',
            n_solutions=100,
            as_ensemble=False,
            n_levels=5,
            early_stopping=False,
            n_children=2,
            allow_synergies=False,
            n_jobs=-1,
        ):

        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array")
        
        # Basic version assumes a single output
        if len(y.shape) != 1:
            raise ValueError("y must be a 1D array")
        
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of rows")
        
        if n_children < 2:
            raise ValueError("n_children must be at least 2")

        if n_jobs == -1:
            n_jobs = os.cpu_count()

        if isinstance(n_solutions, list):
            if len(n_solutions) != n_levels:
                raise ValueError("n_solutions must be a list of length n_levels")

        self.n_jobs = n_jobs
        self.n_children = n_children
        self.allow_synergies = allow_synergies
        self.early_stopping = early_stopping
        self.n_solutions = n_solutions
        self.as_ensemble = as_ensemble
        self.n_levels = n_levels
        self.metric = metric
        self.n_cols = X.shape[1]
        self.solutions = []
        self.shift_amounts = X.min(axis=0)
        self.X = self._prepare_X(X)
        self.y = y

    def _prepare_X(self, X):
        if X.shape[1] != self.n_cols:
            raise ValueError(f"X must have the same number of columns as the original X ({self.n_cols})")        
        X = self._shift_columns(X)
        X = self.add_primitives(X)
        return X

    def _shift_columns(self, X):
        if X.shape[1] != self.n_cols:
            raise ValueError(f"X must have the same number of columns as the original X ({self.n_cols})")        
        return X - self.shift_amounts + 1e-8

    def add_primitives(self, X):
        for i in range(X.shape[1]):
            for fn in [np.log, np.sqrt, np.exp, lambda x: 1 / x, lambda x: x ** 2]:
                x = fn(X[:, i]).reshape(-1, 1)
                X = np.hstack((X, x))

        if self.allow_synergies:
            for i1, i2 in combinations(range(self.n_cols), 2):
                for fn in [lambda x1, x2: x1 * x2, lambda x1, x2: x1 / x2]:
                    x = fn(X[:, i1], X[:, i2]).reshape(-1, 1)
                    X = np.hstack((X, x))

        return X
    
    def _init_primitive_population(self):
        self.primitive_population = []
        fitnesses = []
        for i in range(self.X.shape[1]):
            mask = np.zeros(self.X.shape[1], dtype=bool)
            mask[i] = True
            node = PrimitiveNode(mask)
            node.fit(self.X, self.y)
            fitness = node.evaluate(self.X, self.y, self.metric)
            self.primitive_population.append(node)
            fitnesses.append(fitness)
        i = np.argmin(fitnesses)
        self.best_fitness = fitnesses[i]
        self.best_model = self.primitive_population[i]

    def _create_next_population(self, population):
        idx = np.arange(len(population))
        new_population = []
        for combination in combinations(idx, self.n_children):
            children = []
            for c in combination:
                children.append(population[c])
            new_population.append(CrossNode(children))
        return new_population

    @staticmethod
    def _fit(population, X, y, metric, queue, i):
        print(f'start process {i}')
        fitnesses = []
        for node in population:
            node.fit(X, y)
            fitness = node.evaluate(X, y, metric)
            fitnesses.append(fitness)
        queue.put((population, fitnesses))

    def _multiprocess_fit(self, population):
        num_jobs = self.n_jobs
        jobs = []
        queues = [Queue() for _ in range(num_jobs)]

        for i in range(num_jobs):
            chunk = population[i::num_jobs]
            p = Process(
                target=self._fit,
                args=(chunk, self.X, self.y, self.metric, queues[i], i)
            )
            jobs.append(p)
            p.start()
        
        population, fitnesses = [], []
        for i, job in enumerate(jobs):
            p, f = queues[i].get()
            population.extend(p)
            fitnesses.extend(f)
            job.join()
        
        return population, fitnesses
    
    def _selection(self, population, fitnesses, level):
        # 20% of the solutions will be chosen (min 1) will be chosen from 
        n_weak_solutions = max(1, int(0.15*(self.n_solutions[level] if isinstance(self.n_solutions, list) else self.n_solutions)))
        n_strong_solutions = (self.n_solutions[level] if isinstance(self.n_solutions, list) else self.n_solutions) - n_weak_solutions

        if n_weak_solutions + n_strong_solutions != self.n_solutions[level]:
            raise ValueError("n_weak_solutions + n_strong_solutions must be equal to n_solutions for level, something went wrong")
        
        i = np.argsort(fitnesses)
        p_strong = [population[j] for j in i[:n_strong_solutions]]
        f_strong = [fitnesses[j] for j in i[:n_strong_solutions]]

        # Preserve some weak solutions
        i_weak = np.random.choice(i[n_strong_solutions:], size=n_weak_solutions, replace=False)
        p_weak = [population[j] for j in i_weak]
        f_weak = [fitnesses[j] for j in i_weak]

        return p_strong + p_weak, f_strong + f_weak

    def fit(self):
        self._init_primitive_population()
        population = self.primitive_population
        for level in range(self.n_levels):
            st = time.time()
            population = self._create_next_population(population)
            population, fitnesses = self._multiprocess_fit(population)
            population, fitnesses = self._selection(population, fitnesses, level)

            if self.early_stopping and fitnesses[0] >= self.best_fitness:
                self.is_fitted = True
                return

            print(f"Level {level} (in {time.time() - st:.2f} s) prev best fitness: {self.best_fitness}, new best fitness: {fitnesses[0]}, % change: {((fitnesses[0] - self.best_fitness) / self.best_fitness) * 100}")
            self.best_fitness = fitnesses[0]
            self.best_model = population[0]

        self.is_fitted = True

    def predict(self, X):
        if not hasattr(self, 'is_fitted'):
            raise ValueError("Model is not fitted")
        X = self._prepare_X(X)
        return self.best_model(X)
    
    def evaluate(self, X, y, metric=None):
        X = self._prepare_X(X)
        return self._evaluate(X, y, self.best_model, metric)
    
    def _evaluate(self, X, y, model, metric):
        metric = self.metric or metric
        y_pred = model(X)
        return METRIC_FNS[metric](y, y_pred)