using MLJ, Random, CSV, DataFrames, CategoricalArrays
using RDatasets
using Printf
using Statistics
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
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

# https://alan-turing-institute.github.io/MLJ.jl/dev/models/KNNClassifier_NearestNeighborModels/#KNNClassifier_NearestNeighborModels
# KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
# X, y = @load_crabs; ## a table and a vector from the crabs dataset
# ## view possible kernels
# NearestNeighborModels.list_kernels()
# ## KNNClassifier instantiation
# model = KNNClassifier(weights = NearestNeighborModels.Inverse())
# mach = machine(model, X, y) |> MLJ.fit! ## wrap model and required data in an MLJ machine and fit
# y_hat = MLJ.predict(mach, X)
# labels = predict_mode(mach, X)

# Input reconfiguration data
recon = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\dataset\\Training set of best configurations.csv",DataFrame,types=Dict(1=>Float64))
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

## view possible kernels
NearestNeighborModels.list_kernels()
## KNNClassifier instantiation
model = KNNClassifier(weights = NearestNeighborModels.Zavreal())
mach = machine(model, Xscaled, y)  
MLJ.fit!(mach, rows=trainRows);
fitted_params(mach)
# # Save the trained machine
# # Using MJL
# MLJ.save("KNN_Zavreal.jl",mach)
# mach_predict_only = machine("KNN_Zavreal.jl")
# MLJ.predict(mach_predict_only, testscaled)
# # Using an arbitrary serializer
# using JLSO
# # This machine can now be serialized
# smach = serializable(mach)
# JLSO.save("KNN_Zavreal.jlso", :machine => smach)

# # Deserialize and restore learned parameters to useable form:
# loaded_mach = JLSO.load("KNN_Zavreal.jlso")[:machine]
# restore!(loaded_mach)
# MLJ.predict(loaded_mach, testscaled)




y_hat = MLJ.predict(mach, testscaled)
labels = predict_mode(mach, testscaled)
a = levelcode.(labels)
findall(a->a==5,a)
df_output = DataFrame(Tin=300, xBset=0.11, T1initial=388.7, T2initial=388.7, T3initial=388.7, xB1initial=0.11, xB2initial=0.11, xB3initial=0.11, xBtinitial=0.11, BestConfiguration="Parallel", PredictedBestConfiguration="Parallel")

for i in 1:length(y[testRows])
    println(y[i + trainRows[end]], labels[i])
    if a[i] == 1
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], y[i + trainRows[end]], "hybrid"))
    elseif a[i] == 2
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], y[i + trainRows[end]], "mixing"))
    elseif a[i] == 3
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], y[i + trainRows[end]], "parallel"))
    else
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], y[i + trainRows[end]], "series"))
    end
end

y_hat_compare = df_output.PredictedBestConfiguration[2:end]
# @printf "Accuracy: %.2f%%\n" mean(y_hat .== y[testRows]) * 100
for i in 1:length(y[testRows])
    println(y[i + trainRows[end]], labels[i])
end
accuracy = mean(y_hat_compare .== y[testRows]) * 100

CSV.write("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\dataset\\KNN_Zavreal_kernel_93.56.csv",df_output)
