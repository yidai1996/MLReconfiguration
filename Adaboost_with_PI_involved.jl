using StatsBase

function adaboost(X, Y, T)
    # Initialize weights
    N = length(Y) # number of data points
    w = fill(1/N, N)
    
    classifiers = []
    alphas = []
    
    for t = 1:T # T boosting rounds in total, is also the number of weak learners
        # Train a weak classifier on the data
        h = train_weak_classifier(X, Y, w)
        
        # Calculate the error
        error = sum([w[i] for i in 1:N if h(X[i]) != Y[i]]) / sum(w)
        
        # Calculate alpha
        alpha = 0.5 * log((1 - error) / error)
        
        # Update weights
        for i in 1:N
            if h(X[i]) == Y[i]
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
        output = sign(sum([alphas[i] * classifiers[i](x) for i in 1:T]))
        return output    
    end
    
    return H
end