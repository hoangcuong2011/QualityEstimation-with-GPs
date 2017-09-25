# QualityEstimation-with-GPs

Dropout: 0.5
batch_size = 128
epochs = 500
model.compile(optimizer=Adam(1e-4), loss=loss)

gp = GP(hyp={
                'lik': np.log(0.3),
                'mean': [],
                'cov': [[0.5], [1.0]],
            },
            inf='infGrid', dlik='dlikGrid',
            opt={'cg_maxit': 20000, 'cg_tol': 1e-4},
            mean='meanZero', cov='covSEiso',
            update_grid=1,
            grid_kwargs={'eq': 1, 'k': 70.},
            batch_size=batch_size,
            nb_train_samples=nb_train_samples)
    outputs = [gp(hidden)]
    return Model(inputs=inputs, outputs=outputs)

----------------------------------------------

Results:

Test RMSE


Network: 512 - 1

Baseline: 0.603218145523

Our: 0.247205531803

Deep Network: 512-512-1

Baseline: 0.273539712177

Our: 0.229833459531

Very Deep Network: 512-512-512-1

Baseline: 0.269470231901

Our: 0.228834157746


Test RMSE: 17.9197881485 - BASELINE - 50 iterations - Networks: 512-512-1

Test RMSE: 17.7667294786 - OUR - 500 iterations - Networks: 512-512-1-GPs

Test RMSE: 17.8275642043 - OUR - 50 iterations - Networks: 512-512-1-GPs - Trained from Baseline 50 as check points

Test RMSE: 17.7436869178 - baseline - 500 iterations - Networks: 512-512-1- using normalization by deviding 150

Test RMSE: 17.9540099251 - baseline - 500 iterations - Networks: 512-512-2-1- using normalization by deviding 150

Test RMSE: 19.1539551824 - Our - 500 iterations - Networks: 512-512-2-1-Gaussian using normalization by deviding 150

----------------------
experiments 23 Sept 2017

baseline: Test RMSE: 0.172326471031
our: 10 iterations: Test RMSE: 0.178795830403 - pseudo inputs: 500
our: 50 iterations: Test RMSE: 0.172099986995 - pseudo inputs: 500

our: 500 iterations: Test RMSE: Test RMSE: 0.174414185187 - pseudo inputs 500 - from scratch

our: 100 iterations: Test RMSE: 0.172564653068 - pseudo inputs: 500
our: 100 iterations: Test RMSE: 0.176254250821 - pseudo inputs: 500 - instead of 5 -> using 10 in algorithm 2 for minibatch


our: 50 iterations: Test RMSE: 0.172149085512 - pseudo inputs: 500


our: 50 iterations: Test RMSE: 0.18063654052 - pseudo inputs: 500 -> minibatch: 32 - failed toan tap

our: 100 iterations: Test RMSE: 0.172278005018 - pseudo inputs: 500

our: 500 iterations: Test RMSE: 0.173728301087 - pseudo inputs: 500

our: 500 iterations: Test RMSE: 0.172191393769 - pseudo inputs: 500

our: 5000 iterations: Test RMSE: 0.219003467042 - pseudo inputs: 500


Some lessons:



1. over-training (i.e. having too many iterations) hurts the performance. 50 seems ok
2. having too many pseudo inputs is really expensive, yet does not help much. 500 seems to be OK.
3. using the initial weights trained from NN seems pretty helpful.
4. 128 seems a good mini-batch number
5. 5 seems a good iterations regarding to algortihm 2 (gp_n_iter)

-- for the baseline:

training NN with many iterations seems pretty helpful: 
5 itertaions:  RMSE: 18.3
50 iterations: RMSE: 0.173979286729
250 iterations: RMSE: 0.171972987861

------ 25 sept 2017 -----

baseline: 512-512-2-1 Test RMSE: 0.171972987861 - 250 iterations

our: 512-512-2-GP and train from scratch: test RMSE: 0.31018739957 ( failed toan tap) - 50 iterations - pseudo inputs: 500





Different kernel functions do not contribute such a significant difference in result, at least from my experiment with quality estimation.

I used validdata.txt as training dataset

maternity kernel: 0.18460638297956458
RBF: 0.1837610677100123

neural network (baseline: 512-512-1): 0.190945125175 - both 500-50 iterations
neural network (baseline: 128-128-1): 0.187025368196 - 500 iterations




