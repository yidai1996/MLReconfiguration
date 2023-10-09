using CSV, DataFrames

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


# df = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\Tree_max_depth_3_68.44_sorted.csv",DataFrame,types=Dict(1=>Float64))
df = CSV.read("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\Adaboost.csv",DataFrame,types=Dict(1=>Float64))
Predict = df.PredictedBestConfiguration
Best = df.BestConfiguration
row_number = nrow(df)
predict_index = zeros(row_number);
best_index = zeros(row_number);
Predict_pi = zeros(row_number);
Best_pi = zeros(row_number);
df_score = df[:,[:parallel, :hybrid, :mixing, :series]]
pi_allconfiguration = Matrix(df_score)
sorted_configuration = Matrix(df[:,[:BestConfiguration, :SecondBestConfiguration, :ThirdBestConfiguration, :WorstBestConfiguration]])

# If need to know how best the configuration is (best, second best, third best, worst), use the following code
# sorted_configuration = Matrix(df[:,[:BestConfiguration, :SecondBestConfiguration, :ThirdBestConfiguration, :WorstBestConfiguration]])
# for i in 1:row_number
#     position = findall(x-> x == Predict[i], sorted_configuration[i,:])
#     println(position)
# end

for i in 1:row_number
    position = findall(x-> x == Predict[i], sorted_configuration[i,:])
    a = sorted_configuration[i,position]
    println(i)
    b = a[1]
    predict_index[i] = String_to_Number(b)
    if predict_index[i] == 5.0
        println("ERROR!")
        
        break
    end
    best_index[i] = String_to_Number(Best[i])
    println(b)

    if ismissing(pi_allconfiguration[i, convert(Int, predict_index[i])])
        pi_allconfiguration[i, convert(Int, predict_index[i])] = 1e10
        println(typeof(pi_allconfiguration[i, convert(Int, predict_index[i])]))
    else 
    end

    Predict_pi[i] = pi_allconfiguration[i, convert(Int, predict_index[i])]
    Best_pi[i] = pi_allconfiguration[i, convert(Int, best_index[i])]
end
# position = findall(x-> x == Predict[3], sorted_configuration[3,:])
# a = sorted_configuration[3,position]
# String_to_Number(a)
# println(predict_index[3])

for i in 1:row_number
    println(Predict_pi[i], " ", Best_pi[i])
end

bbb = exp.(100*(- Predict_pi + Best_pi)./Best_pi)

score1 = sum(bbb)/row_number

# index_smallscore = findall(x -> x < 1 && x > 0.9, bbb)
index_smallscore = findall(x -> x < 0.8, bbb)

for i in 1:length(index_smallscore)
    println(Predict_pi[index_smallscore[i]]," ", Best_pi[index_smallscore[i]])
end

test1 = Predict_pi[index_smallscore]
test2 = Best_pi[index_smallscore]
highscore_butnotSecondBest = findall(x-> x>2, predict_score2[index_smallscore])
for i in 1:length(highscore_butnotSecondBest)
    println(test1[highscore_butnotSecondBest[i]]," ", test2[highscore_butnotSecondBest[i]])
end


present_smallcase_score = bbb[index_smallscore]
present_smallcase_pi = Predict_pi[index_smallscore]
present_smallcase_best_pi = Best_pi[index_smallscore]

# If need to know how best the configuration is (best, second best, third best, worst), use the following code
sorted_configuration2 = Matrix(df[:,[:BestConfiguration, :SecondBestConfiguration, :ThirdBestConfiguration, :WorstBestConfiguration]])
predict_score2 = zeros(row_number);
best_score2 = zeros(row_number);
for i in 1:row_number
    predict_score2[i] = findall(x-> x == Predict[i], sorted_configuration2[i,:])[1]
    best_score2[i] = findall(x-> x == Best[i], sorted_configuration2[i,:])[1]
end
# test_predict = df.Prediction_numpy
# test_target = df.Target
# element_comparsion = isequal.(Predict,Best)
# different_cases = findall(x -> x == 0, element_comparsion)
# element_comparsion = isequal.(test_predict, test_target)
# different_cases = findall(x -> x == 0, element_comparsion)

element_comparsion = isequal.(predict_score2, best_score2)
different_cases = findall(x -> x == 0, element_comparsion)
different_predict = predict_score2[different_cases]
different_number = length(different_predict)
second_best_number = length(findall(x -> x == 2, different_predict))
third_best_number = length(findall(x -> x == 3, different_predict))
worst_number =  length(findall(x -> x == 4, different_predict))

(row_number - third_best_number - worst_number - second_best_number)/row_number

# PI Percentage of wrong prediction
# Prediction | BestConfiguration | PI of Prediction | PI of BestConfiguration | delta PI/ PI of BestConfiguration
percentage = zeros(row_number);
for i in 1:row_number
    percentage[i] = (Predict_pi[i] - Best_pi[i])/Best_pi[i]
end
df_percentage = df[:,[:PredictedBestConfiguration, :BestConfiguration]]
df_percentage[!, :Predict_PI] = Predict_pi
df_percentage[!, :Best_PI] = Best_pi
df_percentage[!, :Percentage] = percentage
CSV.write("C:\\Users\\yid\\TemporaryResearchDataStorage\\Reconfiguration\\Space_filling_sampling\\dataset\\score\\score of Adaboost.csv",df_percentage)

different_in_percentage = percentage[findall(x-> x>0, percentage)]
greatthan_10 = percentage[findall(x-> x>0.1, percentage)]
findmax(different_in_percentages)