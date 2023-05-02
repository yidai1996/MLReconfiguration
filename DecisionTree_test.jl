using MLJ, Random, CSV, DataFrames, CategoricalArrays
using RDatasets
using Printf
using Statistics
using DecisionTree
Tree = @load DecisionTreeClassifier pkg=DecisionTree
# doc("DecisionTreeClassifier", pkg="DecisionTree")
# referencevector is if we want to scale with reference to the training data
# for normal scaling, just use v twice
function scaleminmax(v, referencevector, minvalue, maxvalue)
    minoriginal = minimum(referencevector)
    maxoriginal = maximum(referencevector)
    newVector = minvalue .+ ((v .- minoriginal) .* (maxvalue - minvalue)) ./
        (maxoriginal - minoriginal)
    return newVector
end

# # https://github.com/JuliaAI/DecisionTree.jl
# features, labels = load_data("iris")
# features = float.(features)
# labels = string.(labels)

# # train depth-truncated classifier
# model = DecisionTreeClassifier(max_depth=2)
# DecisionTree.fit!(model, features, labels)
# # pretty print of the tree, to a depth of 5 nodes (optional)
# print_tree(model, 5)
# # apply learned model
# DecisionTree.predict(model, [5.9,3.0,5.1,1.9])
# # get the probability of each label
# predict_proba(model, [5.9,3.0,5.1,1.9])
# println(get_classes(model)) # returns the ordering of the columns in predict_proba's output
# # run n-fold cross validation over 3 CV folds
# # See ScikitLearn.jl for installation instructions
# using ScikitLearn.CrossValidation: cross_val_score
# accuracy = cross_val_score(model, features, labels, cv=3)



# Input reconfiguration data
recon = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\dataset\\Training set of best configurations.csv",DataFrame,types=Dict(1=>Float64))
# recon = CSV.read("G:\\My Drive\\Research\\SVM\\Training dataset\\Initial conditions_setpointtracking_disturbancerejection_permutation\\Second\\Training set with nine features.csv",DataFrame,types=Dict(1=>Float64))
# convert(CategoricalArrays.categorical,recon[:,3])

transform!(recon, names(recon, AbstractString) .=> categorical, renamecols=false)
rng = MersenneTwister(1234);
recon = recon[shuffle(rng, 1:end), :]
trainRows, testRows = partition(eachindex(recon.BestConfiguration), 0.9); # 50:50 split
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

features = float.(Matrix(trainscaled))
labels = string.(ytrain)

# train depth-truncated classifier
model = DecisionTreeClassifier(max_depth=8)
DecisionTree.fit!(model, features, labels)
# pretty print of the tree, to a depth of 5 nodes (optional)
print_tree(model, 8)
# # apply learned model
# DecisionTree.predict(model, [5.9,3.0,5.1,1.9])
# # get the probability of each label
# predict_proba(model, [5.9,3.0,5.1,1.9])
println(get_classes(model)) # returns the ordering of the columns in predict_proba's output
# run n-fold cross validation over 3 CV folds
# See ScikitLearn.jl for installation instructions
using ScikitLearn.CrossValidation: cross_val_score
accuracy = cross_val_score(model, features, labels, cv=3)

y_hat = DecisionTree.predict(model, float.(Matrix(testscaled)))
# df_output = DataFrame(Tin=300, xBset=0.11, T1initial=388.7, T2initial=388.7, T3initial=388.7, xB1initial=0.11, xB2initial=0.11, xB3initial=0.11, xBtinitial=0.11, BestConfiguration="Parallel", PredictedBestConfiguration="Parallel")

for i in 1:length(y[testRows])
    println(y[i + trainRows[end]], y_hat[i])
    # push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], y[i + trainRows[end]], y_hat[i]))
end


# @printf "Accuracy: %.2f%%\n" mean(y_hat .== y[testRows]) * 100
accuracy = mean(y_hat .== y[testRows]) * 100




# TODO calculate the accuracy for each configuration (the first and second dataset) 

# CSV.write("G:\\My Drive\\Research\\SVM\\Training dataset\\Initial conditions_setpointtracking_disturbancerejection_permutation\\Second\\SVM results_70training.csv",df_output)
