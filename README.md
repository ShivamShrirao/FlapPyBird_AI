# FlapPyBird_AI
A simple AI bot to play FlappyBird. It learns over generations. Fitness score is calculated based on distance travelled and points scored. The best of the generation are cloned, crossover, mutated and some random genomes are added to population.
The base game code was taken from https://github.com/f-prime/FlappyBird of just 90 lines. Made a lot of changes to work with neural network, changed some physics, hitbox system, number of pipes, etc.

Acceleration of pipes and gravity can be changed easily.

In non accelerating pipes a perfect fit can cross over 4500 pipes and still not die.
![4000 pipes](/screenshot.png?raw=true "Crossed 4000 pipes")

In accelerating pipes it mostly just dies cause it is physically impossible for it to reach next gap in time.
