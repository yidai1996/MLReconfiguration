using MLJ, RDatasets, Printf, Statistics, Random, DataFrames, CSV, LIBSVM, CategoricalArrays
import LIBSVM

SVC = @load SVC pkg=LIBSVM
# NuSVC = @load NuSVC pkg=LIBSVM

# doc("NuSVC", pkg="LIBSVM")
doc("SVC", pkg="LIBSVM")
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

doc("SVC")


# Input reconfiguration data
recon = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\Training set of best configurations with sorted.csv",DataFrame,types=Dict(1=>Float64))
# recon = CSV.read("G:\\My Drive\\Research\\SVM\\Training dataset\\Initial conditions_setpointtracking_disturbancerejection_permutation\\Second\\Training set with nine features.csv",DataFrame,types=Dict(1=>Float64))
# function SVM_Reconfiguration(recon)
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

# model = svmtrain(train, ytrain)

# For reconfiguration
# model = NuSVC(kernel=LIBSVM.Kernel.RadialBasis)
# model = SVC()
model = SVC(kernel=LIBSVM.Kernel.RadialBasis)
# tuning the model with cross validation and a grid for kernel hyperparameters
# r1 = range(model, :gamma, lower=0.00001 , upper=5, scale=:log)
# r2 = range(model, :cost, lower=1000, upper = 50000, scale=:log)
# r1 = range(model, :gamma, lower=0.00001 , upper=0.5, scale=:log)
# r2 = range(model, :cost, lower=10000, upper = 20000000, scale=:log10)
# resampling is the cross validation parameters 
# TODO figure out what is MisclassificationRate()
# tm = TunedModel(model=model, tuning=Grid(resolution=10),
#                 resampling=CV(nfolds=3), ranges=[r1, r2],
#                 measure=MisclassificationRate())
mach = machine(model, Xscaled,y) 
MLJ.fit!(mach, rows=trainRows);
fitted_params(mach)
r = report(mach)

# bestModel = r.best_model
# bestHistory = r.best_history_entry

y_hat = MLJ.predict(mach, testscaled)
df_output = DataFrame(Tin=300, xBset=0.11, T1initial=388.7, T2initial=388.7, T3initial=388.7, xB1initial=0.11, xB2initial=0.11, xB3initial=0.11, xBtinitial=0.11, BestConfiguration="Parallel", PredictedBestConfiguration="Parallel")

for i in 1:length(y[testRows])
    println(y[i + trainRows[end]], y_hat[i])
    push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], y[i + trainRows[end]], y_hat[i]))
end


# @printf "Accuracy: %.2f%%\n" mean(y_hat .== y[testRows]) * 100
accuracy = mean(y_hat .== y[testRows]) * 100
return accuracy

MLJ.save("SVM.jl",mach)
# mach_predict_only = machine("KNN_Zavreal_best_bigdata.jl")
# MLJ.predict(mach_predict_only, testscaled)

# TODO calculate the accuracy for each configuration (the first and second dataset) 

CSV.write("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\SVM results_70training.csv",df_output)


# Convert targets from string to float
# df_test = CSV.read("G:\\My Drive\\Research\\SVM\\Training dataset\\Initial conditions_setpointtracking_disturbancerejection_permutation\\First try 0123\\Training set with nine features.csv", DataFrame)
# row_num = nrow(df_test)
# TargetConfiguration = Float64[]
# best = df_test.BestConfiguration
# for i in 1:row_num
#     if best[i] == "parallel"
#         push!(TargetConfiguration, 1)
#     elseif best[i] == "hybrid"
#         push!(TargetConfiguration, 2)
#     elseif best[i] == "mixing"
#         push!(TargetConfiguration, 3)
#     else
#         push!(TargetConfiguration, 4)
#     end
# end
# df_test[!,:TargetConfiguration] = TargetConfiguration
# CSV.write("G:\\My Drive\\Research\\SVM\\Training dataset\\Initial conditions_setpointtracking_disturbancerejection_permutation\\First try 0123\\Training set with nine features_targets.csv",df_test)
