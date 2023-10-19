# using StatsBase
using MLJ, DataFrames, CSV, Random
using CategoricalArrays
import LIBSVM
SVC = @load SVC pkg=LIBSVM
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
Tree = @load DecisionTreeClassifier pkg=DecisionTree

function String_to_Number(a)
    # println(a)
    c = 0 
    if a == "parallel"
        c = 1.0
    elseif a == "hybrid"
        c = 2.0
    elseif a == "mixing"
        c = 3.0
    elseif a == "series"
        c = 4.0
    else 
        c = 5.0
    end
    return c
end


My_Machines = ["SVM.jl", "KNN_Zavreal_best_bigdata.jl", "Tree_max_depth_9.jl"]
# My_Machines = ["SVM.jl"]
function scaleminmax(v, referencevector, minvalue, maxvalue)
    minoriginal = minimum(referencevector)
    maxoriginal = maximum(referencevector)
    newVector = minvalue .+ ((v .- minoriginal) .* (maxvalue - minvalue)) ./
        (maxoriginal - minoriginal)
    return newVector
end

function load_data(recon)
    # recon = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\additional_data\\PreDataSetForReconfiguration3\\dataset\\Training set of best configurations with sorted.csv",DataFrame,types=Dict(1=>Float64))
    
    transform!(recon, names(recon, AbstractString) .=> categorical, renamecols=false)
    rng = MersenneTwister(1234);
    recon = recon[shuffle(rng, 1:end), :]
    trainRows, testRows = partition(eachindex(recon.BestConfiguration), 0.7); # 70:30 split
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
    return trainscaled, ytrain, testscaled, y[testRows], trainRows, testRows, X, y, recon
end

function NewDataframe(h, y, X, recon)
    df_output = DataFrame(Tin=300, xBset=0.11, T1initial=388.7, T2initial=388.7, T3initial=388.7, xB1initial=0.11, xB2initial=0.11, xB3initial=0.11, xBtinitial=0.11, parallel=0.0, hybrid=0.0, mixing=0.0, series=0.0, BestConfiguration="parallel", SecondBestConfiguration="hybrid", ThirdBestConfiguration="mixing", WorstBestConfiguration="series", PredictedBestConfiguration="Parallel")
    for i in eachindex(y[testRows])
        println(y[i + trainRows[end]], h[i])
        if h[i] == "hybrid"
            push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "hybrid"))
        elseif h[i] == "mixing"
            push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "mixing"))
        elseif h[i] == "parallel"
            push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "parallel"))
        else
            push!(df_output,(X[i + trainRows[end], 1], X[i + trainRows[end], 2], X[i + trainRows[end], 3], X[i + trainRows[end], 4], X[i + trainRows[end], 5], X[i + trainRows[end], 6], X[i + trainRows[end], 7], X[i + trainRows[end], 8], X[i + trainRows[end], 9], recon[i + trainRows[end],"parallel"], recon[i + trainRows[end],"hybrid"], recon[i + trainRows[end],"mixing"], recon[i + trainRows[end],"series"], recon[i + trainRows[end],"BestConfiguration"], recon[i + trainRows[end],"SecondBestConfiguration"], recon[i + trainRows[end],"ThirdBestConfiguration"], recon[i + trainRows[end],"WorstBestConfiguration"], "series"))
        end
    end
    df_output = delete!(df_output, [1])
    
    return df_output
end

# New error calculation test
function Score(h, df)
    count = 0
    Predict = df.PredictedBestConfiguration
    Best = df.BestConfiguration
    row_number = size(h)[1]
    predict_index = zeros(row_number);
    best_index = zeros(row_number);
    max_index = zeros(row_number);
    Predict_pi = zeros(row_number);
    Best_pi = zeros(row_number);
    Max_pi = zeros(row_number);
    df_score = df[:,[:parallel, :hybrid, :mixing, :series]]
    pi_allconfiguration = Matrix(df_score)
    percentage = zeros(row_number);
    # Replace NaN performance index with 1e10
    for i in 1:row_number
        for j in 1:4
            if isnan(pi_allconfiguration[i, j])
                # println(typeof(pi_allconfiguration[i, j]))
                pi_allconfiguration[i, j] = 1e10
                
            else 
            end
        end
    end
    sorted_configuration = Matrix(df[:,[:BestConfiguration, :SecondBestConfiguration, :ThirdBestConfiguration, :WorstBestConfiguration]])
    for i in 1:row_number
        position = findall(x-> x == Predict[i], sorted_configuration[i,:])
        a = sorted_configuration[i,position]
        b = a[1]
        predict_index[i] = String_to_Number(b)
        if predict_index[i] == 5.0
            println("ERROR!")
        
            break
        end
        best_index[i] = String_to_Number(Best[i])

        Predict_pi[i] = pi_allconfiguration[i, convert(Int, predict_index[i])]
        Best_pi[i] = pi_allconfiguration[i, convert(Int, best_index[i])]
        max_index[i] = findmax(pi_allconfiguration[i,:])[2]
        # println(max_index[i])
        Max_pi[i] = pi_allconfiguration[i, convert(Int, max_index[i])]
        if Predict[i] == Best[i]
            count +=1
        end
        percentage[i] = (Predict_pi[i] - Best_pi[i])/Best_pi[i]

    end
    score = (Predict_pi-Best_pi)./(Max_pi-Best_pi)
    return score, Predict_pi, Best_pi, percentage, count
end

# test = predict_mode(machine("KNN_Zavreal_best_bigdata_new.jl"),XX)
# WriteIntoFiles(test;filename="real test")
# new = NewDataframe(test, y, X, recon)
# ss, A, B, C, cc = Score(test, new)

function adaboost(X, Y, My_Machines)
    # Initialize weights
    N = length(Y) # number of data points
    # println(N)
    w = fill(1/N, N)
    
    classifiers = []
    alphas = []
    mach = []
    h = []
    for i in 1:1
    for t in 1:length(My_Machines) # T boosting rounds in total, is also the number of weak learners
        # Train a weak classifier on the data
        println(My_Machines[t])
        push!(mach, machine(My_Machines[t]))
        if t == 1
            h = MLJ.predict(mach[t], X)
        else
            h = predict_mode(mach[t], X)
        end
        println(size(h))
        # h = train_weak_classifier(X, Y, w)
        # calculate score that is related to performance index
        new_df = NewDataframe(h, recon)
        s = Score(h, new_df)

        # Calculate the error
        error = sum(w[j]*s[j] for j in eachindex(w) if h[j]!=Y[j]) / sum(w.*s)
        # error = sum(w[j] for j in eachindex(w) if h[j]!=Y[j]) / sum(w)
        println(error)

        if error > 0.5
            println("ERROR > 0.5 ABORT")
            continue
        end

        # Calculate alpha
        alpha = log((1 - error) / error) 
        # alpha = log((1 - error) / error) 
        println(alpha)
        # Update weights
        # for i in 1:N
        #     if h[i] == Y[i]
        #         w[i] = w[i] * exp(-alpha)
        #     else
        #         w[i] = w[i] * exp(alpha)
        #     end
        # end
        for i in 1:N
            if h[i] == Y[i]
                w[i] = w[i] * exp(-alpha) * 5*s[i]
            else
                w[i] = w[i] * exp(alpha) * 5*s[i]
            end
        end

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
function finalH(a, x, classifiers)

    h1 = MLJ.predict(classifiers[1],x)
    h2 = predict_mode(classifiers[2],x)
    h3 = predict_mode(classifiers[3],x)

    WriteIntoFiles(h1; filename = "Member_SVM")
    WriteIntoFiles(h2; filename = "Member_KNN")
    WriteIntoFiles(h3; filename = "Member_DT")

    println(size(a)[1])
    if size(a)[1] == 3
        h = [h1 h2 h3]
    else
        h = [h1 h2 h3 h1 h2 h3]
    end
    T1 = zeros(size(x)[1],length(a))
    T1[findall(x->x=="parallel", h)] .= 1
    AlphaMatrix = zeros(size(x)[1],4)
    println(a)
    AlphaMatrix[:,1] = T1*a
    T2 = zeros(size(x)[1],length(a))
    T2[findall(x->x=="hybrid", h)] .= 1
    AlphaMatrix[:,2] = T2*a
    T3 = zeros(size(x)[1],length(a))
    T3[findall(x->x=="mixing", h)] .= 1
    AlphaMatrix[:,3] = T3*a
    T4 = zeros(size(x)[1],length(a))
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
df_original = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\Training set of best configurations with sorted.csv",DataFrame,types=Dict(1=>Float64))

XX, YY, X_test, Y_test, trainRows, testRows, X, y, recon = load_data(recon)

y_hat = finalH(a, X_test, adab_classifier)
accuracy = mean(Y_test .== y_hat) * 100


function WriteIntoFiles(y_hat; filename = "Notspecific")
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
        # println(i)
    end
    # println(df_output)

    df_output = delete!(df_output, [1])
    s, A, B, C, D = Score(y_hat,df_output)
    df_output[!,:Score] = s
    # println(mean(Y_test .== y_hat) * 100)
    count = 0
    for i in eachindex(s)
        if s[i] == 0
            count += 1
        end
    end
    accuracy = count/size(y_hat)[1]
    println("accuracy = ",accuracy)
    CSV.write("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\"*filename*".csv",df_output)
end



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


mach=machine("KNN_Zavreal_best_bigdata_new.jl")
# y_hat = MLJ.predict(mach, X_test)
labels = predict_mode(mach, X_test)
# a = levelcode.(labels)
# findall(a->a==5,a)
df_output = DataFrame(Tin=300, xBset=0.11, T1initial=388.7, T2initial=388.7, T3initial=388.7, xB1initial=0.11, xB2initial=0.11, xB3initial=0.11, xBtinitial=0.11, parallel=0.0, hybrid=0.0, mixing=0.0, series=0.0, BestConfiguration="parallel", SecondBestConfiguration="hybrid", ThirdBestConfiguration="mixing", WorstBestConfiguration="series", PredictedBestConfiguration="parallel")
# TODO This doesn't give the right dataframe file!!!
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

y_hat_compare = df_output.PredictedBestConfiguration[2:end]
Y_testtest = df_output.BestConfiguration[2:end]
# @printf "Accuracy: %.2f%%\n" mean(y_hat_compare .== Y_test) * 100
for i in 1:length(Y_testtest)
    println("From y")
    println(y[i + trainRows[end]], labels[i])
    println("From Y_testtest")
    println(Y_testtest[i], labels[i])
end
accuracy = mean(y_hat_compare .== Y_test) * 100
df_output = delete!(df_output, [1])
dfff = CSV.write("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\KNN_Zavreal_kernel_new_sorted.csv",df_output)

ss, A, B, C, D= Score(y_hat_compare, df_output)