# SpectralDensities.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://quantum-exeter.github.io/SpectralDensities.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://quantum-exeter.github.io/SpectralDensities.jl/dev/)
[![Build Status](https://github.com/quantum-exeter/SpectralDensities.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/quantum-exeter/SpectralDensities.jl/actions/workflows/CI.yml?query=branch%3Amain)

SpectralDensities.jl is a package that defines commonly used spectral densities for Open Quantum Systems and typical operations on them.

## Installation
To install the latest stable release of SpectralDensities.jl, you can use Julia's built-in package manager.
From the Julia interpreter press the `]` key and run
```Julia
pkg> add SpectralDensities
```
Alternatively, you can import the Pkg package and call `add` from it
```Julia
using Pkg;
Pkg.add("SpectralDensities")
```
