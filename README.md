# denison-2023
The code I produced as part of a research project into Quantum Computing. I didn't get very far on the quantum side.

The 3 Versions of the simulator are:
- [exact_qho.py](https://github.com/simo-bimo/denison-2023/blob/f14ae70c614a930bd52f92f020982b6ee3bca980/exact_qho.py)
- [soft_simulator.py](https://github.com/simo-bimo/denison-2023/blob/f14ae70c614a930bd52f92f020982b6ee3bca980/soft_simulator.py)
- [soft_simulator_3d.py](https://github.com/simo-bimo/denison-2023/blob/f14ae70c614a930bd52f92f020982b6ee3bca980/soft_simulator_3d.py)

## Exact QHO

The exact_qho was my first piece of code, it's not very good. It computes the exact solution of a quantum harmonic oscillator. i.e. You switch into a Fock basis (of the QHO solutions) and evolve each basis state in time, then switch back. It's animations are a bit laggy, and not very good.

## Soft Simulator

Soft Simulator is the best one, as it uses the Split Operator Fourier Transform method. It is implemented as a soft_simulator class, with a series of functional tests. The default values are alright, you generate an animation by calling the `animate_pos` or `animate_pos_momentum` functions. They each return a `MatPlotLib` animation object, which you can play back with `plt.show()` (this renders every frame as you play back, it's quite slow), or save it as a `.gif` with `anim.save('file_location.gif', 'pillow', fps=30)`. 

If you recode it, it recommend you keep the animation/rendering component a separate function (as it gives you much more flexibility), like I did for Varied Params. Being part of the object is fine for static plots.

See below for a smooth animation:

![Soft Simulator Animation](https://github.com/simo-bimo/denison-2023/blob/f14ae70c614a930bd52f92f020982b6ee3bca980/Soft-Simulator-Animations/SOFT%20Fat%20Offset%20Gaussian.gif)

### Vary Params

The Vary Params part is a little complicated, it generates a series of different simulator objects, and animates all of them simultaneously on the same animation (so it's easier to compare what changes in different values are). See Below:

![Vary Parameters Animation](https://github.com/simo-bimo/denison-2023/blob/f14ae70c614a930bd52f92f020982b6ee3bca980/Varied%20Parameters%20for%20Coulomb/Vary%20Vmax%20Centred.gif)

### 3D Soft Simulator

The 3D version of the soft simulator was a bit of a rushed job, and it's not worth very much because Matplot lib is bad at animating 3D things, and you have to re-render the entire plot everytime (rather than update the points). You can update the points, but only their positions, you can't change the number of points plotted (which you need to do if you're plotting some 3D function above a certain constant). It certainly looks cool though:

![A 3D Animation of a QHO](https://github.com/simo-bimo/denison-2023/blob/f14ae70c614a930bd52f92f020982b6ee3bca980/3D%20Animations/QHO_test_HQ.gif)

## QFT Implementation

This is a little thing I looked at on my own, because I wanted to do more quantum algorithms, but 6 weeks isn't quite enough time to really get into that.

## A small sidenote

I have explicitly type-hinted as much as I can (everything that isn't a pain in python anyway), which is what all the `Callable[[np.ndarray], np.ndarray]]` stuff is. That means 'this argument should be a function which takes a numpy array as input and gives a numpy array as output'. This is generally just good programming practice, and though I know it's annoying it helps you avoid many stupid errors (and sometimes forces you to write better logic).

Good luck! Have fun. :)

P.S. There's also a TODO list in there, which serves as a bit of a log for the whole project.
