using CategoricalArrays, MixedModels, Random, JuMP, Ipopt, LinearAlgebra, Statistics, DataFrames, CSV, Distributions, GLM, StatsBase, SparseArrays

function create_data_cluster_unfair_lr(np, nc, seed)
    s = seed

    kkkk1 = np

    β = [-20; 4; 8; 5; 30] ./ 10

    sexH = rand(MersenneTwister(42), kkkk1, 1) .< 0.5
    b = rand(MersenneTwister(42), Normal(0, 3), 100)
    mean = [0.0, 0.0, 0.0]
    C = [1.0 0 0; 0 1.0 0; 0 0 1.0]
    d = MvNormal(mean, C)
    X = [ones(kkkk1) rand(MersenneTwister(42), d, kkkk1)' sexH]

    N = nc
    function predicao34(β, X, b, N)

        m12(x, β, b) = 1 / (1 + exp(-dot(β, x) - b))
        NSC2 = 1000 * ones(100)

        ly = size(X)[1]
        verificar = zeros(ly)
        y = zeros(ly)
        k = 1
        cara = zeros(N)
        for i = 1:N
            cara[i] = sum(NSC2[j] for j = 1:i)
        end

        for i = 1:ly
            if i > cara[1]
                k += 1
                cara = cara[2:end]
            end
            verificar[i] = m12(X[i, :], β, b[k])
            d = [Binomial(1, verificar[i])]
            y[i] = rand(MersenneTwister(42), d[1], 1)[1]
        end
        return y
    end
    y = predicao34(β, X, b, N)

    target = zeros(kkkk1)
    for i = 1:kkkk1
        if y[i] == 1
            target[i] = 1
        else
            target[i] = 0
        end
    end

    sex = sexH
    CDT = [X[:, 2:end-1] sex target]

    CDT = DataFrame(CDT, :auto)
    CDT = rename(CDT, :x1 => :F1, :x2 => :F2, :x3 => :F3, :x4 => :Sex, :x5 => :target)

    contador1 = 0
    contador2 = 0
    for i = 1:kkkk1
        if CDT.Sex[i] == 0
            contador1 += 1
            if CDT.target[i] == 1
                contador2 += 1
            end
        end
    end

    k, l = size(CDT)
    NN = kkkk1
    CDT1 = CDT[:, 1:5]

    k, l = size(CDT1)
    N = nc
    SC = round(Int64(k / N))

    z = zeros(NN)
    for i = 1:N
        z[(i-1)*SC+1:i*SC] = i * ones(SC)
    end

    CDT1 = [categorical(z) CDT1]
    CDT1 = rename(CDT1, :x1 => :ClusterID)

    Y = CDT[:, 5]
    Y = [y == 1 ? 1 : 0 for y in Y]

    medias = zeros(N)
    for i = 1:N
        medias[i] = Statistics.mean(CDT1.target[(i-1)*SC+1:i*SC])
    end

    medias_til = medias / (sum(medias))

    nct = rand(MersenneTwister(42), 3:5, N)
    nct = Int64.(nct)

    snct = Int64(sum(nct))
    c1 = zeros(snct)
    c1[1:nct[1]] = sort(sample(MersenneTwister(s), ((1-1)*SC+1):(SC*1), nct[1], replace=false))
    for i = 2:length(nct)# N
        c1[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] = sort(sample(MersenneTwister(s), ((i-1)*SC+1):(SC*i), nct[i], replace=false))
    end
    c1
    X = CDT1

    I_train = []
    for i in c1
        I_train = [I_train; i]
    end

    I_train = Int64.(I_train)
    I_test = deleteat!(collect(1:k), sort(I_train))

    X_train = X[I_train, :]
    Y_train = Y[I_train]

    X_test = X[I_test, :]
    Y_test = Y[I_test]

    X = X_train
    Y = Y_train

    return X_train, Y_train, nct
end

# Generalized Linear Mixed Models
function bGLMM(X_train, Y_train, nct)
    X = X_train
    Y = Y_train
    slp = X
    k, p = size(X)
    n = length(nct)

    verbaggform = @formula(target ~ 1 + (1 | ClusterID) + F1 + F2 + F3 + Sex)
    gm1 = fit(MixedModel, verbaggform, slp, Binomial())
    nobs(gm1)
    η = GLM.predict(gm1, slp; type=:linpred)
    μ = GLM.fitted(gm1)
    b = ranef(gm1)[1]'

    Q = 3

    X = Matrix(X[:, 1:end-1])
    nx, p = size(X)
    Z = zeros(nx, n)
    for j = 1:n
        for i = 1:nx
            if X[i, 1] == j
                Z[i, j] = 1
            end
        end
    end
    X = Matrix(X[:, 2:end])
    nx, p = size(X)
    β0 = 1
    β = zeros(p)

    A = [ones(k) X Z]

    Q_1 = inv(Q)
    t = sparse(Q_1 * I, 1, 1)
    for i = 2:n
        t = blockdiag(t, sparse(Q_1 * I, 1, 1))
    end
    K = blockdiag(sparse(0I, 2, 2), t)
    K_1 = blockdiag(sparse(0I, 1, 1), t)
    A_1 = [ones(k) Z]

    δ_r = zeros(p, 2 + n)
    for r = 1:p
        δ_r[r, :] = [β0; β[r]; b]
    end

    D = zeros(k, k)
    BIC = zeros(p)
    Vb = zeros(n)
    F_bb = zeros(p, p)
    s = zeros(p, p)
    F_i = zeros(n)
    F_b = zeros(n, p)
    s = zeros(p, p)
    ϵ = 0

    for i = 1:k
        D[i, i] = GLM.mueta(LogitLink(), η[i])
        if D[i, i] ≤ 1e-4 || isnan(D[i, i])
            D[i, i] = 0.0001
        end
    end
    Σ = copy(D)
    Σ2 = sqrt.(Σ)

    W_l = copy(D)
    W_1 = W_l

    M_0 = A_1 * (A_1' * W_1 * A_1 + K_1) * A_1' * W_1
    prodI = I - M_0
    W_l

    for r = 1:p
        X_r = [ones(nx) X[:, r]]
        A_r = [X_r Z]

        F_r = A_r' * W_l * A_r + K
        s_r = A_r' * (Y - μ) - K * δ_r[r, :]

        δ_r[r, :] = F_r \ s_r

        Σ2 = sqrt.(Σ)
        W_l2 = sqrt.(W_l)
        H_til_r = W_l2 * A_r * ((F_r) \ A_r') * W_l2

        M_r = Σ2 * (H_til_r / Σ2)

        prodI = prodI * (I - M_r)
        H_r = I - prodI

        l_μ_r1 = sum(Y[t] * log(μ[t]) + (1 - Y[t]) * log(1 - μ[t]) for t = 1:nct[1])
        l_μ_r = sum(sum(Y[sum(nct[j] for j = 1:i-1)+t] * log(μ[sum(nct[j] for j = 1:i-1)+t]) + (1 - Y[sum(nct[j] for j = 1:i-1)+t]) * log(1 - μ[sum(nct[j] for j = 1:i-1)+t]) for t = 1:nct[i]) for i = 2:n)
        l_μ_r = l_μ_r + l_μ_r1

        BIC[r] = -2 * l_μ_r + 2 * tr(H_r) * log(n)
    end

    j1 = argmin(BIC)[1]
    δ_j = δ_r[j1, :]

    β0 += δ_j[1]
    b += δ_j[3:end]
    β[j1] += δ_j[2]


    δ = [β0; β; b]
    η = A * δ
    μ = GLM.linkinv.(LogitLink(), η)

    F_bb1 = X[1:nct[1], :]' * D[1:nct[1], 1:nct[1]] * Σ2[1:nct[1], 1:nct[1]] * D[1:nct[1], 1:nct[1]] * X[1:nct[1], :]

    for i = 2:n
        F_bb += X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]' * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Σ2[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]
    end
    F_bb = F_bb + F_bb1

    Z_1 = Z[1:nct[1], 1]
    F_i[1] = Z_1' * D[1, 1] * Σ2[1, 1] * D[1, 1] * Z_1 + inv(Q)
    F_b[1, :] = X[1:nct[1], :]' * D[1:nct[1], 1:nct[1]] * Σ2[1:nct[1], 1:nct[1]] * D[1:nct[1], 1:nct[1]] * Z_1
    for i = 2:n
        Z_i = Z[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], i]
        F_i[i] = Z_i' * D[i, i] * Σ2[i, i] * D[i, i] * Z_i + inv(Q)
        F_b[i, :] = X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]' * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Σ2[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Z_i
    end

    for i = 1:n
        s += F_b[i, :] * F_i[i]^(-1) * F_b[i, :]'
    end

    for i = 1:n
        V_ii = F_i[i]^(-1) + F_i[i]^(-1) * F_b[i, :]' * (F_bb - s)^(-1) * F_b[i, :] * F_i[i]^(-1)
        Vb[i] = V_ii + b[i]^2
    end

    Q_old = Q
    Q = Statistics.mean(Vb)

    for l = 2:3000
        for i = 1:k

            D[i, i] = GLM.mueta(LogitLink(), η[i])
            if D[i, i] ≤ 1e-4 || isnan(D[i, i])
                D[i, i] = 0.0001
            end
        end
        Σ = copy(D)
        Σ2 = sqrt.(Σ)
        W_l = copy(D)

        for r = 1:p
            X_r = [ones(nx) X[:, r]]
            A_r = [X_r Z]

            F_r = A_r' * W_l * A_r + K
            s_r = A_r' * W_l * (D \ (Y - μ)) - K * δ_r[r, :]

            δ_r[r, :] = F_r \ s_r

            Σ2 = sqrt.(Σ)
            W_l2 = sqrt.(W_l)
            H_til_r = W_l2 * A_r * ((F_r) \ A_r') * W_l2

            M_r = Σ2 * (H_til_r / Σ2)

            prodI = prodI * (I - M_r)
            H_r = I - prodI

            l_μ_r1 = sum(Y[t] * log(μ[t]) + (1 - Y[t]) * log(1 - μ[t]) for t = 1:nct[1])
            l_μ_r = sum(sum(Y[sum(nct[j] for j = 1:i-1)+t] * log(μ[sum(nct[j] for j = 1:i-1)+t]) + (1 - Y[sum(nct[j] for j = 1:i-1)+t]) * log(1 - μ[sum(nct[j] for j = 1:i-1)+t]) for t = 1:nct[i]) for i = 2:n)
            l_μ_r = l_μ_r + l_μ_r1

            BIC[r] = -2 * l_μ_r + 2 * tr(H_r) * log(n)
        end

        j1 = argmin(BIC)[1]
        δ_j = δ_r[j1, :]

        β0 += δ_j[1]
        b += δ_j[3:end]
        β[j1] += δ_j[2]


        δ = [β0; β; b]
        η = A * δ
        μ = GLM.linkinv.(LogitLink(), η)

        F_bb1 = X[1:nct[1], :]' * D[1:nct[1], 1:nct[1]] * Σ2[1:nct[1], 1:nct[1]] * D[1:nct[1], 1:nct[1]] * X[1:nct[1], :]

        for i = 2:n
            F_bb += X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]' * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Σ2[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]
        end
        F_bb = F_bb + F_bb1

        Z_1 = Z[1:nct[1], 1]
        F_i[1] = Z_1' * D[1, 1] * Σ2[1, 1] * D[1, 1] * Z_1 + inv(Q)
        F_b[1, :] = X[1:nct[1], :]' * D[1:nct[1], 1:nct[1]] * Σ2[1:nct[1], 1:nct[1]] * D[1:nct[1], 1:nct[1]] * Z_1
        for i = 2:n
            Z_i = Z[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], i]
            F_i[i] = Z_i' * D[i, i] * Σ2[i, i] * D[i, i] * Z_i + inv(Q)
            F_b[i, :] = X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]' * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Σ2[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Z_i
        end

        for i = 1:n
            s += F_b[i, :] * F_i[i]^(-1) * F_b[i, :]'
        end

        for i = 1:n
            V_ii = F_i[i]^(-1) + F_i[i]^(-1) * F_b[i, :]' * (F_bb - s)^(-1) * F_b[i, :] * F_i[i]^(-1)
            Vb[i] = V_ii + b[i]^2
        end

        Q_old = Q
        Q = Statistics.mean(Vb)
        ϵ = abs((Q_old - Q) / Q)

        if ϵ ≤ 0.001
            break
        end
        println("Iteration, $l")
    end

    β0f = δ[1]
    βf = δ[2:p+1]
    bf = δ[p+2:end]

    return β0f, βf, bf, ϵ
end

# Fair Generalized Linear Mixed Models
function bGLMM_Fair(X_train, Y_train, nct, SF, c1, ρ)
    X = X_train
    Y = Y_train
    Zin = SF
    peso_c = 0
    slp = X
    k, p = size(X)
    n = length(nct)

    η = zeros(k)
    μ = zeros(k)
    Q = 3

    X = Matrix(X[:, 1:end-1])
    nx, p = size(X)

    Z = zeros(nx, n)
    for j = 1:n
        for i = 1:nx
            if X[i, 1] == j
                Z[i, j] = 1
            end
        end
    end
    X = Matrix(X[:, 2:end])
    nx, p = size(X)
    kkk1, kkk2 = size(X_train)

    Z0 = [Zin]

    predfixF, predrandF = di_me_logreg([ones(kkk1) X_train[:, 2:end-1]], Y_train, Z0, 0.1, nct)
    β = predfixF[2:end]
    β0 = predfixF[1]
    b = predrandF


    A = [ones(k) X Z]
    Xones = [ones(k) X]
    Q_1 = inv(Q)
    t = sparse(Q_1 * I, 1, 1)
    for i = 2:n
        t = blockdiag(t, sparse(Q_1 * I, 1, 1))
    end
    K = blockdiag(sparse(0I, 2, 2), t)
    K_1 = blockdiag(sparse(0I, 1, 1), t)
    A_1 = [ones(k) Z]

    δ_r = zeros(p, 2 + n)
    for r = 1:p
        δ_r[r, :] = [β0; β[r]; b]
    end

    Z0 = Z0[1]
    Z0 = X_train[:, Z0]
    D = zeros(k, k)
    BIC = zeros(p)
    Vb = zeros(n)
    F_bb = zeros(p, p)
    s = zeros(p, p)
    F_i = zeros(n)
    F_b = zeros(n, p)
    s = zeros(p, p)
    ϵ = 0
    meansz = zeros(n)
    zeh = zeros(k)

    for i = 1:n
        meansz[i] = Statistics.mean(Z0)
    end

    zeh[1:nct[1]] = Z0[1:nct[1]] .- meansz[1]
    for i = 2:n
        zeh[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] = Z0[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] .- meansz[i]
    end

    Form_A = sum((zeh .* Xones), dims=1)

    Form_A2 = zeros(n)
    Form_A2[1] = sum(zeh[1:nct[1]])
    for i = 2:n
        Form_A2[i] = sum(zeh[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]])
    end

    A_Newton = [Form_A Form_A2']

    for i = 1:k
        D[i, i] = GLM.mueta(LogitLink(), η[i])
        if D[i, i] ≤ 1e-4 || isnan(D[i, i])
            D[i, i] = 0.0001
        end
    end
    Σ = copy(D)
    Σ2 = sqrt.(Σ)
    W_l = copy(D)
    W_1 = W_l

    M_0 = A_1 * (A_1' * W_1 * A_1 + K_1) * A_1' * W_1

    prodI = I - M_0

    CandMR = zeros(k, k, p)

    for r = 1:p
        X_r = [ones(nx) X[:, r]]
        A_r = [X_r Z]

        F_r = A_r' * W_l * A_r + K
        s_r = A_r' * (Y - μ) - K * δ_r[r, :]

        A_Newton_r = [A_Newton[:, 1] A_Newton[:, r+1] A_Newton[:, p+2:end]]

        δ_r[r, :] = (F_r + ρ * A_Newton_r' * A_Newton_r) \ (s_r - ρ * A_Newton_r' * ((A_Newton_r*δ_r[r, :])[1] - peso_c * k))

        Σ2 = sqrt.(Σ)
        W_l2 = sqrt.(W_l)
        H_til_r = W_l2 * A_r * ((F_r + ρ * A_Newton_r' * A_Newton_r) \ A_r') * W_l2

        M_r = Σ2 * (H_til_r / Σ2)
        CandMR[:, :, r] = M_r
        H_r = I - prodI * (I - M_r)

        l_μ_r1 = sum(Y[t] * log(μ[t]) + (1 - Y[t]) * log(1 - μ[t]) for t = 1:nct[1])
        l_μ_r = sum(sum(Y[sum(nct[j] for j = 1:i-1)+t] * log(μ[sum(nct[j] for j = 1:i-1)+t]) + (1 - Y[sum(nct[j] for j = 1:i-1)+t]) * log(1 - μ[sum(nct[j] for j = 1:i-1)+t]) for t = 1:nct[i]) for i = 2:n)
        l_μ_r = l_μ_r + l_μ_r1 - ρ * δ_r[r, :]' * A_Newton_r' * A_Newton_r * δ_r[r, :]

        BIC[r] = -2 * l_μ_r + 2 * tr(H_r) * log(n)

    end

    j1 = argmin(BIC)[1]
    δ_j = δ_r[j1, :]

    β0 += δ_j[1]
    b += δ_j[3:end]
    β[j1] += δ_j[2]

    δ = [β0; β; b]

    prodI = (I - CandMR[:, :, j1]) * prodI

    η = A * δ
    μ = GLM.linkinv.(LogitLink(), η)

    F_bb1 = X[1:nct[1], :]' * D[1:nct[1], 1:nct[1]] * Σ2[1:nct[1], 1:nct[1]] * D[1:nct[1], 1:nct[1]] * X[1:nct[1], :]
    for i = 2:n
        F_bb += X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]' * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Σ2[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]
    end
    F_bb = F_bb + F_bb1


    Z_1 = Z[1:nct[1], 1]
    F_i[1] = Z_1' * D[1, 1] * Σ2[1, 1] * D[1, 1] * Z_1 + inv(Q)
    F_b[1, :] = X[1:nct[1], :]' * D[1:nct[1], 1:nct[1]] * Σ2[1:nct[1], 1:nct[1]] * D[1:nct[1], 1:nct[1]] * Z_1
    for i = 2:n
        Z_i = Z[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], i]
        F_i[i] = Z_i' * D[i, i] * Σ2[i, i] * D[i, i] * Z_i + inv(Q)
        F_b[i, :] = X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]' * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Σ2[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Z_i
    end

    for i = 1:n
        s += F_b[i, :] * F_i[i]^(-1) * F_b[i, :]'
    end

    for i = 1:n
        V_ii = F_i[i]^(-1) + F_i[i]^(-1) * F_b[i, :]' * (F_bb - s)^(-1) * F_b[i, :] * F_i[i]^(-1)
        Vb[i] = V_ii + b[i]^2
    end

    Q_old = Q
    Q = Statistics.mean(Vb)

    for l = 2:3000
        for i = 1:k
            D[i, i] = GLM.mueta(LogitLink(), η[i])
            if D[i, i] ≤ 1e-4 || isnan(D[i, i])
                D[i, i] = 0.0001
            end
        end
        Σ = copy(D)
        Σ2 = sqrt.(Σ)

        W_l = copy(D)

        CandMR = zeros(k, k, p)
        for r = 1:p
            X_r = [ones(nx) X[:, r]]
            A_r = [X_r Z]

            F_r = A_r' * W_l * A_r + K
            s_r = A_r' * W_l * (D \ (Y - μ)) - K * δ_r[r, :]

            A_Newton_r = [A_Newton[:, 1] A_Newton[:, r+1] A_Newton[:, p+2:end]]

            δ_r[r, :] = (F_r + ρ * A_Newton_r' * A_Newton_r) \ (s_r - ρ * A_Newton_r' * ((A_Newton_r*δ_r[r, :])[1] - peso_c * k))





            Σ2 = sqrt.(Σ)
            W_l2 = sqrt.(W_l)
            H_til_r = W_l2 * A_r * ((F_r + ρ * A_Newton_r' * A_Newton_r) \ A_r') * W_l2

            M_r = Σ2 * (H_til_r / Σ2)
            CandMR[:, :, r] = M_r
            H_r = I - prodI * (I - M_r)

            l_μ_r1 = sum(Y[t] * log(μ[t]) + (1 - Y[t]) * log(1 - μ[t]) for t = 1:nct[1])
            l_μ_r = sum(sum(Y[sum(nct[j] for j = 1:i-1)+t] * log(μ[sum(nct[j] for j = 1:i-1)+t]) + (1 - Y[sum(nct[j] for j = 1:i-1)+t]) * log(1 - μ[sum(nct[j] for j = 1:i-1)+t]) for t = 1:nct[i]) for i = 2:n)
            l_μ_r = l_μ_r + l_μ_r1 - ρ * δ_r[r, :]' * A_Newton_r' * A_Newton_r * δ_r[r, :]

            BIC[r] = -2 * l_μ_r + 2 * tr(H_r) * log(n)
        end

        j1 = argmin(BIC)[1]
        δ_j = δ_r[j1, :]

        β0 += δ_j[1]
        b += δ_j[3:end]
        β[j1] += δ_j[2]


        δ = [β0; β; b]

        prodI = (I - CandMR[:, :, j1]) * prodI

        η = A * δ
        μ = GLM.linkinv.(LogitLink(), η)

        F_bb1 = X[1:nct[1], :]' * D[1:nct[1], 1:nct[1]] * Σ2[1:nct[1], 1:nct[1]] * D[1:nct[1], 1:nct[1]] * X[1:nct[1], :]

        for i = 2:n
            F_bb += X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]' * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Σ2[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]
        end
        F_bb = F_bb + F_bb1

        Z_1 = Z[1:nct[1], 1]
        F_i[1] = Z_1' * D[1, 1] * Σ2[1, 1] * D[1, 1] * Z_1 + inv(Q)
        F_b[1, :] = X[1:nct[1], :]' * D[1:nct[1], 1:nct[1]] * Σ2[1:nct[1], 1:nct[1]] * D[1:nct[1], 1:nct[1]] * Z_1
        for i = 2:n
            Z_i = Z[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], i]
            F_i[i] = Z_i' * D[i, i] * Σ2[i, i] * D[i, i] * Z_i + inv(Q)
            F_b[i, :] = X[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], :]' * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Σ2[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * D[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i]] * Z_i
        end

        for i = 1:n
            s += F_b[i, :] * F_i[i]^(-1) * F_b[i, :]'
        end

        for i = 1:n
            V_ii = F_i[i]^(-1) + F_i[i]^(-1) * F_b[i, :]' * (F_bb - s)^(-1) * F_b[i, :] * F_i[i]^(-1)
            Vb[i] = V_ii + b[i]^2
        end

        Q_old = Q
        Q = Statistics.mean(Vb)
        ϵ = abs((Q_old - Q) / Q)

        if ϵ ≤ 0.001
            break
        end
        println("Iteration, $l")
    end

    β0f = δ[1]
    βf = δ[2:p+1]
    bf = δ[p+2:end]

    return β0f, βf, bf, ϵ
end

# Cluster Regularized Logistic Regression (CRLR)
function id_me_logreg(X_train, Y_train, nct)
    nc = length(nct)
    m, n = size(X_train)
    Y_train = [y == 1 ? 1 : -1 for y in Y_train]

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, b[1:nc])

    @NLobjective(model, Min, -sum(((Y_train[i] + 1) / 2) * log(1 / (1 + exp(-(sum(β[k] * X_train[i, k] for k = 1:n) + b[1])))) + ((1 - Y_train[i]) / 2) * log(1 - 1 / (1 + exp(-(sum(β[k] * X_train[i, k] for k = 1:n) + b[1])))) for i = 1:nct[1]) - sum(sum(((Y_train[i] + 1) / 2) * log(1 / (1 + exp(-(sum(β[k] * X_train[i, k] for k = 1:n) + b[l])))) + ((1 - Y_train[i]) / 2) * log(1 - 1 / (1 + exp(-(sum(β[k] * X_train[i, k] for k = 1:n) + b[l])))) for i = sum(nct[r] for r = 1:l-1)+1:sum(nct[r] for r = 1:l-1)+nct[l]) for l = 2:nc) + sum(b[i]^2 for i = 1:nc))
    optimize!(model)

    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    return PredFix, PredRand
end

# Fair Cluster Regularized Logistic Regression (Fair CRLR)
function di_me_logreg(X_train, Y_train, SF, c, nct)
    Z0 = SF
    nc = length(nct)
    m, n = size(X_train)
    Y_train = [y == 1 ? 1 : -1 for y in Y_train]

    Z = Matrix(X_train[!, Z0])
    means_z = Statistics.mean(Z, dims=1)

    zeh = zeros(m, length(Z0))
    for p = 1:length(Z0)
        zeh[1:nct[1], p] = Z[1:nct[1], p] .- means_z[1, p]
    end
    for p = 1:length(Z0)
        for i = 2:nc
            zeh[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], p] = Z[sum(nct[j] for j = 1:i-1)+1:sum(nct[j] for j = 1:i-1)+nct[i], p] .- means_z[1, p]
        end
    end

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    set_time_limit_sec(model, 60.0)
    @variable(model, β[1:n])
    @variable(model, b[1:nc])

    @NLobjective(model, Min, -sum(((Y_train[i] + 1) / 2) * log(1 / (1 + exp(-(sum(β[k] * X_train[i, k] for k = 1:n) + b[1])))) + ((1 - Y_train[i]) / 2) * log(1 - 1 / (1 + exp(-(sum(β[k] * X_train[i, k] for k = 1:n) + b[1])))) for i = 1:nct[1]) - sum(sum(((Y_train[i] + 1) / 2) * log(1 / (1 + exp(-(sum(β[k] * X_train[i, k] for k = 1:n) + b[l])))) + ((1 - Y_train[i]) / 2) * log(1 - 1 / (1 + exp(-(sum(β[k] * X_train[i, k] for k = 1:n) + b[l])))) for i = sum(nct[r] for r = 1:l-1)+1:sum(nct[r] for r = 1:l-1)+nct[l]) for l = 2:nc) + sum(b[i]^2 for i = 1:nc))
    @constraint(model, [k = 1:size(Z, 2)], ((1 / m) * sum(zeh[j, k] * (dot(β, X_train[j, :]) + b[1]) for j = 1:nct[1]) + (1 / m) * sum(sum(zeh[j, k] * (dot(β, X_train[j, :]) + b[i]) for j = sum(nct[r] for r = 1:i-1)+1:sum(nct[r] for r = 1:i-1)+nct[i]) for i = 2:nc)) ≤ c)
    @constraint(model, [k = 1:size(Z, 2)], ((1 / m) * sum(zeh[j, k] * (dot(β, X_train[j, :]) + b[1]) for j = 1:nct[1]) + (1 / m) * sum(sum(zeh[j, k] * (dot(β, X_train[j, :]) + b[i]) for j = sum(nct[r] for r = 1:i-1)+1:sum(nct[r] for r = 1:i-1)+nct[i]) for i = 2:nc)) ≥ -c)

    optimize!(model)

    PredFix, PredRand = JuMP.value.(β), JuMP.value.(b)

    return PredFix, PredRand
end





# nct is the size of each cluster

X_train, Y_train, nct = create_data_cluster_unfair_lr(100000, 100, 42)
SF = 5     # Sensitive Feature
ρ = 0.8    # Lagrange Multiplier from Fair BGLMM
c = 0.1    # Fairness threshold from Fair CRLR

predfix, predrand = id_me_logreg([ones(size(X_train, 1)) X_train[:, 2:end-1]], Y_train, nct)
predfixF, predrandF = di_me_logreg([ones(size(X_train, 1)) X_train[:, 2:end-1]], Y_train, [SF], c, nct)
β0f, βf, bf, ϵ = bGLMM(X_train, Y_train, nct)
β0fF, βfF, bfF, ϵF = bGLMM_Fair(X_train, Y_train, nct, SF, c, ρ)
