# using StatsBase
using MLJ, DataFrames, CSV, Random
import LIBSVM
SVC = @load SVC pkg=LIBSVM
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
Tree = @load DecisionTreeClassifier pkg=DecisionTree

My_Machines = ["SVM.jl", "KNN_Zavreal_best_bigdata.jl", "Tree_max_depth_9.jl"]
# My_Machines = ["SVM.jl"]
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
    return trainscaled, ytrain, testscaled, y[testRows], trainRows, testRows, X, y
end

XX, YY, X_test, Y_test, trainRows, testRows, X, y = load_data()
function adaboost(X, Y, My_Machines)
    # Initialize weights
    N = length(Y) # number of data points
    w = fill(1/N, N)
    
    classifiers = []
    alphas = []
    mach = []
    h = []
    for i in 1:2
    for t in 1:length(My_Machines) # T boosting rounds in total, is also the number of weak learners
        # Train a weak classifier on the data
        println(My_Machines[t])
        push!(mach, machine(My_Machines[t]))
        if t == 1
            h = MLJ.predict(mach[t], X)
        else
            h = predict_mode(mach[t], X)
        end
        # h = train_weak_classifier(X, Y, w)
        # TODO looks like push!(h,h) doesn't make sense
        # Calculate the error
        error = sum(w[j] for j in eachindex(w) if h[j]!=Y[j]) / sum(w)
        println(error)

        if error > 0.5
            println("ERROR > 0.5 ABORT")
            continue
        end

        # Calculate alpha
        alpha = log((1 - error) / error) + log(4-1)
        # alpha = log((1 - error) / error) 
        println(alpha)
        # Update weights
        for i in 1:N
            if h[i] == Y[i]
                w[i] = w[i] * exp(-alpha)
            else
                w[i] = w[i] * exp(alpha)
            end
        end
        # println(w)
        # Normalize weights
        w = w / sum(w)
        # println("This is the new w!")
        # println(w)
        push!(classifiers, mach[t])
        push!(alphas, alpha)
    end
    end
    
    # Final classifier
    function H(x)
        h = []
        for i in 1:3
            if i == 1
                hh = MLJ.predict(machine(My_Machines[1]),x)
            else
                hh = predict_mode(mach[i],x)
            end
            push!(h,hh)
        end
        println(size(h))
        # Four categories
        # for i in size(x)[1]
        #     # parallel
        #     c1 = alphas[i]*h
        # output = sign(sum([alphas[i] * h[i] for i in 1:3]))
        # return output    
    end
    
    return alphas, mach, w, H
end

a, adab_classifier, weight, H = adaboost(XX,YY, My_Machines)
# Final classifier
function finalH(a, x, My_Machines)
    h1 = MLJ.predict(machine(My_Machines[1]),x)
    h2 = predict_mode(machine(My_Machines[2]),x)
    h3 = predict_mode(machine(My_Machines[3]),x)
    h = [h1 h2 h3]
    T1 = zeros(size(x)[1],length(My_Machines))
    T1[findall(x->x=="parallel", h)] .= 1
    AlphaMatrix = zeros(size(x)[1],4)
    AlphaMatrix[:,1] = T1*a
    T2 = zeros(size(x)[1],length(My_Machines))
    T2[findall(x->x=="hybrid", h)] .= 1
    AlphaMatrix[:,2] = T2*a
    T3 = zeros(size(x)[1],length(My_Machines))
    T3[findall(x->x=="mixing", h)] .= 1
    AlphaMatrix[:,3] = T3*a
    T4 = zeros(size(x)[1],length(My_Machines))
    T4[findall(x->x=="series", h)] .= 1
    AlphaMatrix[:,4] = T4*a
    Max_index = findmax(AlphaMatrix, dims=2)[2]
    Prediction_number = zeros(size(x)[1])
    output = String15[]
    for i in eachindex(Max_index)
        Prediction_number[i] = Max_index[i][2]
        if Prediction_number[i] == 1
            conf = "parallel" 
        elseif Prediction_number[i] == 2
            conf = "hybrid"
        elseif Prediction_number[i] == 3
            conf = "mixing"
        else 
            conf = "series"
        end
        push!(output, conf)
    end

    return output
end
# test
y_hat = finalH(a, X_test, My_Machines)
accuracy = mean(Y_test .== y_hat) * 100

df_output = DataFrame(Tin=300, xBset=0.11, T1initial=388.7, T2initial=388.7, T3initial=388.7, xB1initial=0.11, xB2initial=0.11, xB3initial=0.11, xBtinitial=0.11, parallel=0.0, hybrid=0.0, mixing=0.0, series=0.0, BestConfiguration="parallel", SecondBestConfiguration="hybrid", ThirdBestConfiguration="mixing", WorstBestConfiguration="series", PredictedBestConfiguration="Parallel")

for i in eachindex(Y_test)
    println(Y_test[i], y_hat[i])
    if y_hat[i] == "hybrid"
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "hybrid"))
    elseif y_hat[i] == "mixing"
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "mixing"))
    elseif y_hat[i] == "parallel"
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "parallel"))
    else
        push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "series"))
    end
    println(i)
end

CSV.write("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\Adaboost.csv",df_output)
println(size(df_output))