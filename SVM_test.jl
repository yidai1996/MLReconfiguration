using MLJ
using RDatasets
using Printf
using Statistics
using Random
using DataFrames
using CSV
using LIBSVM
using CategoricalArrays
SVC = @load SVC pkg=LIBSVM

# referencevector is if we want to scale with reference to the training data
# for normal scaling, just use v twice
function scaleminmax(v, referencevector, minvalue, maxvalue)
    minoriginal = minimum(referencevector)
    maxoriginal = maximum(referencevector)
    newVector = minvalue .+ ((v .- minoriginal) .* (maxvalue - minvalue)) ./
        (maxoriginal - minoriginal)
    return newVector
end

# doc("SVC");
# # Load Fisher's classic iris data
# iris = dataset("datasets", "iris")
# iris[:,5]
# rng = MersenneTwister(1234);
# iris = iris[shuffle(rng, 1:end), :]
# # Split the dataset into training set and testing set
# trainRows, testRows = partition(eachindex(iris.Species), 0.5); # 70:30 split

# # First four dimension of input data is features
# # X = iris[:, 1:4]
# X = iris[:, 1:2]
# train = X[trainRows, :]
# test = X[testRows, :]
# trainscaled = deepcopy(train)
# testscaled = deepcopy(test)
# for i in 1:2
#     trainscaled[:, i] = scaleminmax(train[:, i], train[:, i], 0, 1) # InexactError: Int64(0.9230769230769231)
#     testscaled[:, i] = scaleminmax(test[:, i], test[:, i], 0, 1)
# end
# Xscaled = vcat(trainscaled, testscaled)

# # LIBSVM handles multi-class data automatically using a one-against-one strategy
# y = iris.Species

# # Train SVM on half of the data using default parameters. See documentation
# # of svmtrain for options
# model = SVC()
# # model = SVC(kernel=LIBSVM.Kernel.Linear)
# # mach = machine(model, X, y)

# # tuning the model with cross validation and a grid for kernel hyperparameters
# r1 = range(model, :gamma, lower=0.00001 , upper=0.5, scale=:log)
# r2 = range(model, :cost, lower=10000, upper = 20000000, scale=:log10)
# # resampling is the cross validation parameters 
# # TODO figure out what is MisclassificationRate()
# tm = TunedModel(model=model, tuning=Grid(resolution=10),
#                 resampling=CV(nfolds=5), ranges=[r1, r2],
#                 measure=MisclassificationRate())
# mach = machine(tm, Xscaled,y)
# MLJ.fit!(mach, rows=trainRows);
# r = report(mach)


# Input reconfiguration data
recon = CSV.read("G:\\My Drive\\Research\\SVM\\Training dataset\\Training set with nine features.csv",DataFrame,types=Dict(1=>Float64))
# convert(CategoricalArrays.categorical,recon[:,3])
transform!(recon, names(recon, AbstractString) .=> categorical, renamecols=false)
rng = MersenneTwister(1234);
recon = recon[shuffle(rng, 1:end), :]
trainRows, testRows = partition(eachindex(recon.BestConfiguration), 0.5); # 50:50 split
# First four dimension of input data is features
X = recon[:, 1:9]
train = X[trainRows, :]
test = X[testRows, :]
trainscaled = deepcopy(train)
testscaled = deepcopy(test)
for i in 1:2
    trainscaled[:, i] = scaleminmax(train[:, i], train[:, i], -1, 1) 
    testscaled[:, i] = scaleminmax(test[:, i], test[:, i], -1, 1)
end
Xscaled = vcat(trainscaled, testscaled)
y = recon.BestConfiguration
ytrain = y[trainRows]

# For reconfiguration
model = SVC(kernel=LIBSVM.Kernel.RadialBasis)
# tuning the model with cross validation and a grid for kernel hyperparameters
r1 = range(model, :gamma, lower=0.00001 , upper=0.5, scale=:log)
r2 = range(model, :cost, lower=10000, upper = 20000000, scale=:log10)
# resampling is the cross validation parameters 
# TODO figure out what is MisclassificationRate()
tm = TunedModel(model=model, tuning=Grid(resolution=10),
                resampling=CV(nfolds=3), ranges=[r1, r2],
                measure=MisclassificationRate())
mach = machine(tm, Xscaled,y)
MLJ.fit!(mach, rows=trainRows);
r = report(mach)

bestModel = r.best_model
bestHistory = r.best_history_entry

y_hat = MLJ.predict(mach, testscaled)
for i in 1:length(y[testRows])
    println(y[i + trainRows[end]], y_hat[i])
end

@printf "Accuracy: %.2f%%\n" mean(y_hat .== y[testRows]) * 100
