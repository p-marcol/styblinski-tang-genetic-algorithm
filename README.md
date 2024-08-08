# Styblinski-Tang function minimization

This repository contains the implementation of the Styblinski-Tang function minimization using genetic algorithm.

## Styblinski-Tang function

The Styblinski-Tang function is a function used as a performance test problem for optimization algorithms. It is defined as follows:
$f(\mathbf{x})=\frac{1}{2}\sum_{i=1}^{n}x_i^4-16x_i^2+5x_i$

where $x_i\in[-5,5]$ for $i=1,2,...,n$.

The global minimum of this function is located at $f(\mathbf{x}^*)=-39.16599n$ for $x_i=-2.903534$ for $i=1,2,...,n$

## Genetic algorithm

Algorithm uses the principles of natural selection to evolve a population of candidate solutions. Best individuals are chosen using tournament selection - best of 3. Crossover is performed using single-point crossover.

---

Project created as part of Biologically Inspired Artificial Intelligence course at Silesian University of Technology.
