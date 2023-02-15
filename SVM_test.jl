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
# Tree = @load DecisionTreeClassifier pkg=DecisionTree

# referencevector is if we want to scale with reference to the training data
# for normal scaling, just use v twice
function scaleminmax(v, referencevector, minvalue, maxvalue)
    minoriginal = minimum(referencevector)
    maxoriginal = maximum(referencevector)
    newVector = minvalue .+ ((v .- minoriginal) .* (maxvalue - minvalue)) ./
        (maxoriginal - minoriginal)
    return newVector
end

doc("SVC");
# # Load Fisher's classic iris data
# iris = dataset("datasets", "iris")
# iris[:,5]
# rng = MersenneTwister(1234);
# iris = iris[shuffle(rng, 1:end), :]
# # Split the dataset into training set and testing set
# trainRows, testRows = partition(eachindex(iris.Species), 0.5); # 70:30 split

# # First four dimension of input data is features
# # X = iris[:, 1:4]
# X = iris[:, 1:4]
# train = X[trainRows, :]
# test = X[testRows, :]
# trainscaled = deepcopy(train)
# testscaled = deepcopy(test)
# for i in 1:4
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
recon = CSV.read("G:\\My Drive\\Research\\SVM\\Training dataset\\Initial conditions_setpointtracking_disturbancerejection_permutation\\First try 0123\\Training set with nine features.csv",DataFrame,types=Dict(1=>Float64))
# recon = CSV.read("G:\\My Drive\\Research\\SVM\\Training dataset\\Initial conditions_setpointtracking_disturbancerejection_permutation\\Second\\Training set with nine features.csv",DataFrame,types=Dict(1=>Float64))
# convert(CategoricalArrays.categorical,recon[:,3])
transform!(recon, names(recon, AbstractString) .=> categorical, renamecols=false)
rng = MersenneTwister(1234);
recon = recon[shuffle(rng, 1:end), :]
trainRows, testRows = partition(eachindex(recon.BestConfiguration), 0.7); # 50:50 split
# First four dimension of input data is features
X = recon[:, 1:9]
train = X[trainRows, :]
test = X[testRows, :]
trainscaled = deepcopy(train)
testscaled = deepcopy(test)

for i in 1:size(X)[2]
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
df_output = DataFrame(Tin=300, xBset=0.11, T1initial=388.7, T2initial=388.7, T3initial=388.7, xB1initial=0.11, xB2initial=0.11, xB3initial=0.11, xBtinitial=0.11, BestConfiguration="Parallel", PredictedBestConfiguration="Parallel")

for i in 1:length(y[testRows])
    println(y[i + trainRows[end]], y_hat[i])
    push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], y[i + trainRows[end]], y_hat[i]))
end


@printf "Accuracy: %.2f%%\n" mean(y_hat .== y[testRows]) * 100

# TODO calculate the accuracy for each configuration (the first and second dataset) 

CSV.write("G:\\My Drive\\Research\\SVM\\Training dataset\\Initial conditions_setpointtracking_disturbancerejection_permutation\\Second\\SVM results_70training.csv",df_output)
