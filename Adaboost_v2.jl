# using StatsBase
using MLJ, DataFrames, CSV, Random
import LIBSVM
SVC = @load SVC pkg=LIBSVM
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
Tree = @load DecisionTreeClassifier pkg=DecisionTree

using PyCall, ScikitLearn
@pyimport sklearn.ensemble as ensemble

function scaleminmax(v, referencevector, minvalue, maxvalue)
    minoriginal = minimum(referencevector)
    maxoriginal = maximum(referencevector)
    newVector = minvalue .+ ((v .- minoriginal) .* (maxvalue - minvalue)) ./
        (maxoriginal - minoriginal)
    return newVector
end
function load_data()
    # recon = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\additional_data\\PreDataSetForReconfiguration3\\dataset\\Training set of best configurations with sorted.csv",DataFrame,types=Dict(1=>Float64))
    recon = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\Training set of best configurations with sorted.csv",DataFrame,types=Dict(1=>Float64))

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
    return trainscaled, ytrain 
end

XX, YY = load_data()
My_Machines = ["SVM.jl", "KNN_Zavreal_best_bigdata.jl", "Tree_max_depth_9.jl"]
mach = []
for t in 1:3 # T boosting rounds in total, is also the number of weak learners
    # Train a weak classifier on the data
    # println(My_Machines[t])
    mach = push!(mach, machine(My_Machines[t]))
end
clf = ensemble.AdaBoostClassifier(estimator = mach)
clf.fit(XX,YY)
clf[:fit](XX, YY)
function adaboost(X, Y, My_Machines)
    # Initialize weights
    N = length(Y) # number of data points
    w = fill(1/N, N)
    
    classifiers = []
    alphas = []
    mach = []

    for t in 1:3 # T boosting rounds in total, is also the number of weak learners
        # Train a weak classifier on the data
        println(My_Machines[t])
        mach = push!(mach, machine(My_Machines[t]))
        h = MLJ.predict(mach[i], X)
        # h = train_weak_classifier(X, Y, w)
        
        # Calculate the error
        error = sum([w[i] for i in 1:N if h[i] != Y[i]]) / sum(w)
        
        # Calculate alpha
        alpha = 0.5 * log((1 - error) / error) + log(4-1)
        
        # Update weights
        for i in 1:N
            if h[i] == Y[i]
                w[i] = w[i] * exp(-alpha)
            else
                w[i] = w[i] * exp(alpha)
            end
        end
        
        # Normalize weights
        w = w / sum(w)
        
        push!(classifiers, h)
        push!(alphas, alpha)
    end
    
    # Final classifier
    function H(x)
        output = sign(sum([alphas[i] * classifiers[i](x) for i in 1:3]))
        return output    
    end
    
    return alphas, classifiers, w
end

a, adab_classifier, weight = adaboost(XX,YY, My_Machines)

# Final classifier
afunction H(x)
    output = sign(sum([alphas[i] * classifiers[i](x) for i in 1:length(x)]))
    return output    
end
# function test(My_Machines)
#     for t in 1:2 # T boosting rounds in total, is also the number of weak learners
#         # Train a weak classifier on the data
#         println(My_Machines[t])
#         mach = machine(My_Machines[2])
#         h = predict_mode(mach,XX)
#     end
# end