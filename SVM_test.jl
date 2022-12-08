
using MLJ
using RDatasets
using Printf
using Statistics
using Random
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

doc("SVC");
# Load Fisher's classic iris data
iris = dataset("datasets", "iris")
rng = MersenneTwister(1234);
# TODO: what does function shuffle() mean
iris = iris[shuffle(rng, 1:end), :]
# Split the dataset into training set and testing set
trainRows, testRows = partition(eachindex(iris.Species), 0.5); # 70:30 split

# First four dimension of input data is features
X = iris[:, 1:4]
train = X[trainRows, :]
test = X[testRows, :]
trainscaled = deepcopy(train)
testscaled = deepcopy(test)
for i in 1:4
    trainscaled[:, i] = scaleminmax(train[:, i], train[:, i], 0, 1)
    testscaled[:, i] = scaleminmax(test[:, i], train[:, i], 0, 1)
end
Xscaled = vcat(trainscaled, testscaled)

# LIBSVM handles multi-class data automatically using a one-against-one strategy
y = iris.Species



# Train SVM on half of the data using default parameters. See documentation
# of svmtrain for options
model = SVC()
# model = SVC(kernel=LIBSVM.Kernel.Linear)
# mach = machine(model, X, y)

# tuning the model with cross validation and a grid for kernel hyperparameters
r1 = range(model, :gamma, lower=0.00001 , upper=0.5, scale=:log)
r2 = range(model, :cost, lower=10000, upper = 20000000, scale=:log10)
# resampling is the cross validation parameters 
tm = TunedModel(model=model, tuning=Grid(resolution=10),
                resampling=CV(nfolds=5), ranges=[r1, r2],
                measure=MisclassificationRate())
mach = machine(tm, Xscaled, y)
fit!(mach, rows=trainRows);
r = report(mach)

bestModel = r.best_model
bestHistory = r.best_history_entry

y_hat = predict(mach, testscaled)
for i in 1:length(y[testRows])
    println(y[i + trainRows[end]], y_hat[i])
end

@printf "Accuracy: %.2f%%\n" mean(y_hat .== y[testRows]) * 100
