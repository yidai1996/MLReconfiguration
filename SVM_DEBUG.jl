using MLJ
import LIBSVM

# PCA = @load PCA pkg=MultivariateStats

# X, y = @load_iris ## a table and a vector

# model = PCA(maxoutdim=2)
# mach = machine(model, X) |> fit!

# Xproj = transform(mach, X)

# SVC = @load SVC pkg=LIBSVM                   ## model type
# model = SVC(kernel=LIBSVM.Kernel.Polynomial) ## instance

# X, y = @load_iris ## table, vector
# mach = machine(model, X, y) |> fit!


# Github bug report of MLJ
#  **Describe the bug**
# <!--
# -->
# I'd like to use LIBSVM and MLJ API on VS Code to do some classification problems. However, the example problem in [SVC](https://alan-turing-institute.github.io/MLJ.jl/dev/models/SVC_LIBSVM/#SVC_LIBSVM) always fails with the following error info:
# The terminal process "C:\Users\username\AppData\Local\Programs\Julia-1.8.5\bin\julia.exe '-i', '--banner=no', '--project=C:\Users\yid\.julia\environments\v1.8', 'c:\Users\username\.vscode\extensions\julialang.language-julia-1.38.2\scripts\terminalserver\terminalserver.jl', '\\.\pipe\vsc-jl-repl-6529904f-c2cf-4632-bc0a-fc20730f91f2', '\\.\pipe\vsc-jl-cr-802319b6-f65b-433d-a80d-54b7933df861', 'USE_REVISE=true', 'USE_PLOTPANE=true', 'USE_PROGRESS=true', 'ENABLE_SHELL_INTEGRATION=true', 'DEBUG_MODE=false'" terminated with exit code: 3221226356.

# I have no idea how to deal with it. 
# Many thanks in advance!
# **To Reproduce**
# <!--
# Add a Minimal, Complete, and Verifiable example (for more details, see e.g. 
# https://stackoverflow.com/help/mcve

# If the code is too long, feel free to put it in a public gist and link
# it in the issue: https://gist.github.com
# -->

# ```julia
# using MLJ
# import LIBSVM

# SVC = @load SVC pkg=LIBSVM                   ## model type
EpsilonSVR = @load EpsilonSVR pkg=LIBSVM                 ## model type
model = EpsilonSVR(kernel=LIBSVM.Kernel.Polynomial)  ## instance
X, y = make_regression(rng=123) ## table, vector
mach = machine(model, X, y) |> fit!

X, y = @load_iris ## table, vector
mach = machine(model, X, y) |> fit!
# ```

# **Expected behavior**
# <!--
# A clear and concise description of what you expected to happen.
# -->
# Hope it can run without termination.
# **Additional context**
# <!--
# Add any other context about the problem here.
# -->

# **Versions**
# <details>
# MLJ 
# <!--
# Please run the following snippet and paste the output here.
# using MLJ
# ...
# -->

# </details>

# <!-- Thanks for contributing! -->


# Conclusion: The problem is https://discourse.julialang.org/t/issue-with-xgboost-jl-and-libsvm-jl-when-julia-1-8-4/92396/19