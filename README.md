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
Baseline: 
Our:
