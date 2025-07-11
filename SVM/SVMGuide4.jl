
using MLJ
using RDatasets
using Printf
using Statistics
using Random
using CSV
using CategoricalArrays
using LIBSVM
SVC = @load SVC pkg=LIBSVM

# scales training and testing data
# scales training data to [minvalue, maxvalue], and then scales
# testing data relative to the scaling up or down that the training
# data underwent
function scaleminmax!(trainingdata, testingdata, minvalue, maxvalue)
    for i in 1:size(trainingdata)[2]
        minoriginal = minimum(trainingdata[:, i])
        maxoriginal = maximum(trainingdata[:, i])
        trainingdata[:, i] = minvalue .+ ((trainingdata[:, i] .- minoriginal)
            .* (maxvalue - minvalue)) ./(maxoriginal - minoriginal)
        testingdata[:, i] = minvalue .+ ((testingdata[:, i] .- minoriginal)
            .* (maxvalue - minvalue)) ./(maxoriginal - minoriginal)
    end
    return [trainingdata, testingdata]
end

doc("SVC")
trainingpath = "C:\\Users\\smafa\\OneDrive - Umich\\Documents\\Allman Work\\svmguide4.txt"
testpath = "C:\\Users\\smafa\\OneDrive - Umich\\Documents\\Allman Work\\svmguide4.t.txt"
train = DataFrame(CSV.File(trainingpath))
test = DataFrame(CSV.File(testpath))


# for converting
# newtrain = DataFrame(Class=deepcopy(train[!,1]))
# newtest = DataFrame(Class=deepcopy(test[!,1]))
# for i in 2:size(train)[2]
#     newcol = Vector{Float64}()
#     for j in 1:size(train)[1]
#         push!(newcol, parse(Float64, split(train[j, i], ":")[2]))
#     end
#     print(newcol)
#     newtrain[!, "Field $(i-1)"] = newcol
# end
#
# for i in 2:size(test)[2]
#     newcol = Vector{Float64}()
#     for j in 1:size(test)[1]
#         push!(newcol, parse(Float64, split(test[j, i], ":")[2]))
#     end
#     print(newcol)
#     newtest[!, "Field $(i-1)"] = newcol
# end
#
#
# CSV.write(trainingpath, newtrain)
# CSV.write(testpath, newtest)


rng = MersenneTwister(1234);
train = train[shuffle(rng, 1:end), :]



trainscaled = deepcopy(train[:, 2:end])
testscaled = deepcopy(test[:, 2:end])
scaleminmax!(trainscaled, testscaled, -1, 1)

X = train[:, 2:end]
Xscaled = trainscaled

# LIBSVM handles multi-class data automatically using a one-against-one strategy
y = CategoricalArray(train.Class)



# Train SVM on half of the data using default parameters. See documentation
# of svmtrain for options
model = SVC()
# mach = machine(model, Xscaled, y)

# tuning the model with cross validation and a grid for kernel hyperparameters
r1 = range(model, :gamma, lower=0.00001 , upper=5, scale=:log)
r2 = range(model, :cost, lower=1000, upper = 50000, scale=:log)
tm = TunedModel(model=model, tuning=Grid(resolution=11),
                resampling=CV(nfolds=5), ranges=[r1, r2],
                measure=MisclassificationRate())
mach = machine(tm, Xscaled, y)
MLJ.fit!(mach)
r = report(mach)

bestModel = r.best_model
bestHistory = r.best_history_entry

y_hat = predict(mach, testscaled)
for i in 1:length(y_hat)
    println(test[i, 1], y_hat[i])
end

@printf "Accuracy: %.2f%%\n" mean(y_hat .== test[:, 1]) * 100

finalmodel = SVC(kernel = LIBSVM.Kernel.RadialBasis, gamma=0.0024298221, cost=32768.0)
