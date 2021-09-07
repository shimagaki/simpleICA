# Input Z is a k_top x pattern_dim matrix
function myICA(Z, n_learn=10_000, l_W=1e-3)
    @show (k_top, pattern_length) = size(Z);
    W=randn(k_top, k_top) * 0.01
    W=copy(W)+copy(W)';
    loss_W_vec = zeros(n_learn);
    for t in 1:n_learn

        WZ = W*Z
        #d_W = (I + (I - (2.0 ./ (1 .+ exp.(WZ) )) * WZ'  ) ) * W  # doesn't work
        d_W = (I - WZ * WZ'  ) * W # works
        #d_W = (I - tanh.(WZ) * WZ' ) * W # works
        W = W + l_W * d_W

        WZ = W*Z
        cov_WZ = WZ * WZ'
        loss_W = norm(I-cov_WZ)
        loss_W_vec[t] = loss_W
        #@show t, loss_W
    end
    return (W, loss_W_vec)
end
