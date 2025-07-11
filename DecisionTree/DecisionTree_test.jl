using MLJ, Random, CSV, DataFrames, CategoricalArrays
using RDatasets
using Printf
using Statistics
# using DecisionTree
# using Pkg
# Pkg.activate("my_fresh_mlj_environment", shared=true)
doc("DecisionTreeClassifier", pkg="DecisionTree")
DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
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
# recon = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\dataset\\Training set of best configurations.csv",DataFrame,types=Dict(1=>Float64))
recon = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\Training set of best configurations with sorted.csv",DataFrame,types=Dict(1=>Float64))

# recon = CSV.read("G:\\My Drive\\Research\\SVM\\Training dataset\\Initial conditions_setpointtracking_disturbancerejection_permutation\\Second\\Training set with nine features.csv",DataFrame,types=Dict(1=>Float64))
# convert(CategoricalArrays.categorical,recon[:,3])

transform!(recon, names(recon, AbstractString) .=> categorical, renamecols=false)
rng = MersenneTwister(1234);
recon = recon[shuffle(rng, 1:end), :]
trainRows, testRows = partition(eachindex(recon.BestConfiguration), 0.7); # 70:30 split
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

# features = float.(Matrix(trainscaled))
# labels = string.(ytrain)

X_tree = (Tin=Xscaled[trainRows,1], xBset=Xscaled[trainRows,2], T1initial=Xscaled[trainRows,3], T2initial=Xscaled[trainRows,4], T3initial=Xscaled[trainRows,5], xB1initial=Xscaled[trainRows,6], xB2initial=Xscaled[trainRows,7], xB3initial=Xscaled[trainRows,8], xBtinitial=Xscaled[trainRows,9])

# train depth-truncated classifier
# X, y=@load_iris
model = DecisionTreeClassifier(max_depth=9)
mach = machine(model, X_tree, ytrain) 
MLJ.fit!(mach)
fitted_params(mach)

r = report(mach)
# pretty print of the tree, to a depth of 5 nodes (optional)
# print_tree(mach, 5)
# # apply learned model
# DecisionTree.predict(model, [5.9,3.0,5.1,1.9])
# # get the probability of each label
# predict_proba(model, [5.9,3.0,5.1,1.9])
# println(get_classes(mach)) # returns the ordering of the columns in predict_proba's output
# run n-fold cross validation over 3 CV folds
# See ScikitLearn.jl for installation instructions
# using ScikitLearn.CrossValidation: cross_val_score
# accuracy = cross_val_score(mach, X_tree, ytrain, cv=3)

# predict_proba(mach,float.(Matrix(testscaled)))
y_hat = MLJ.predict(mach, testscaled)
labels = predict_mode(mach, testscaled)
# df_output = DataFrame(Tin=300, xBset=0.11, T1initial=388.7, T2initial=388.7, T3initial=388.7, xB1initial=0.11, xB2initial=0.11, xB3initial=0.11, xBtinitial=0.11, BestConfiguration="Parallel", PredictedBestConfiguration="Parallel")
df_output = DataFrame(Tin=300, xBset=0.11, T1initial=388.7, T2initial=388.7, T3initial=388.7, xB1initial=0.11, xB2initial=0.11, xB3initial=0.11, xBtinitial=0.11, parallel=0.0, hybrid=0.0, mixing=0.0, series=0.0, BestConfiguration="parallel", SecondBestConfiguration="hybrid", ThirdBestConfiguration="mixing", WorstBestConfiguration="series", PredictedBestConfiguration="Parallel")

for i in 1:length(y[testRows])
    println(y[i + trainRows[end]], y_hat[i])
    # push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], y[i + trainRows[end]], y_hat[i]))
end
for i in 1:length(y[testRows])
    println(y[i + trainRows[end]], labels[i])
    if labels[i] == "hybrid"
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "hybrid"))
    elseif labels[i] == "mixing"
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "mixing"))
    elseif labels[i] == "parallel"
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "parallel"))
    else
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "series"))
    end
end

# @printf "Accuracy: %.2f%%\n" mean(y_hat .== y[testRows]) * 100
accuracy = mean(labels .== y[testRows]) * 100

# # Save the trained machine
using MLJ
MLJ.save("Tree_max_depth_9.jl",mach)
# mach_predict_only = machine("Tree_max_depth_3.jl")
# MLJ.predict(mach_predict_only, testscaled)


# TODO calculate the accuracy for each configuration (the first and second dataset) 

CSV.write("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\Tree_max_depth_9_84.99_sorted.csv",df_output)
