```@meta
CurrentModule = SpectralDensities
```

# SpectralDensities

This package implements commonly used spectral densities for
Open Quantum Systems and typical operations on them.

## Features

Features of SpectralDensities.jl include:
* Definition of most widely used spectral densities:
Ohmic, Sub-Ohmic, Supra-Ohmic, Lorentzian (Underdamped), Debye (Overdamped)
* Flexibility to choose desidred cutoff functions:
hard cutoffs, exponential cutoff, gaussian cutoff.
* Calculation of the correlation function of a given spectral density
(with specialised methods for the pre-defined types)
* Calculation of reorganisation energy of a given spectral density
(with specialised methods for the pre-defined types)
* Methods to compute the spectral density integrals that appear in the
weak coupling (2nd order expansion), both for
the dynamics (see e.g. H. Breuer, F. Petruccione, "The Theory of Open Quantum Systems"),
and the equilibrium mean-force state (see e.g. J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021)).

## Quick start

The desired spectral density can be constructed by simply
passing the desired parameters to their respective constructor.
For example, for an Ohmic spectral density ``J(\\omega) = 3\\omega``
we have
```math
Johmic = OhmicSD(3)
```

To add a ctuoff to a base spectral density, one simply passes it
to the cutoff constructor. For example, to add an exponential cutoff
``e^{-\\omega/7}`` we can do
```math
Jc = ExponentialCutoffSD(Johmic, 7)
```

## Defining custom spectral densities

The base AbstractSD can be easily extended by the user.
To do so, one must create a new sub-type for which *at least* one
of the methods `sd` (the spectral density itself ```math J(\\omega)```)
or `sdoverω` (the spectral density divided by frequency ```math J(\\omega)/\\omega```)
must be defined. It is strongly suggested to define, if possible, `sdoverω` since it
typically makes it simpler to numerically handle potential singularities.

This is the only requirement, and all other methods will
automatically work on the new subtype. Of course, as is typical
of Julia's super flexible multiple-dispatch system, the user should
consider providing custom functions on the new type for the reorganisation energy, etc.