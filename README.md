# Styblinski-Tang function minimization

This repository contains the implementation of the Styblinski-Tang function minimization using genetic algorithm.

## Styblinski-Tang function

The Styblinski-Tang function is a function used as a performance test problem for optimization algorithms. It is defined as follows:

![equation](<https://latex.codecogs.com/gif.latex?f(\mathbf{x})&space;=&space;\frac{1}{2}\sum_{i=1}^{n}x_i^4&space;-&space;16x_i^2&space;+&space;5x_i>)

where ![equation](https://latex.codecogs.com/gif.latex?x_i&space;\in&space;[-5,5]) for ![equation](https://latex.codecogs.com/gif.latex?i&space;=&space;1,2,...,n).

The global minimum of this function is located at ![equation](<https://latex.codecogs.com/gif.latex?f(\mathbf{x}^*)&space;=&space;-39.16599n>) for ![equation](https://latex.codecogs.com/gif.latex?x_i&space;=&space;-2.903534) for ![equation](https://latex.codecogs.com/gif.latex?i&space;=&space;1,2,...,n).

## Genetic algorithm

Algorithm uses the principles of natural selection to evolve a population of candidate solutions. Best individuals are chosen using tournament selection - best of 3. Crossover is performed using single-point crossover.

---

Project created as part of Biologically Inspired Artificial Intelligence course at Silesian University of Technology.
