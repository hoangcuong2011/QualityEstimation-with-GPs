Task to do:


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


---- for GPS

Sparse GPs seem pretty helpful regarding to scalability.

1. Pseudo inducing points = 1 -> 0.18207436990727555

2. Pseudo inducing points = 10 -> 0.17422071640407702

3. Pseudo inducing points = 100 -> 0.17135238660272237

4. Pseudo inducing points = 500 -> 0.17045979565507988

5. Pseudo inducing points = 1000 -> 0.17026061242154458


------ 25 sept 2017 -----

baseline: 512-512-2-1 Test RMSE: 0.171972987861 - 250 iterations

our: 512-512-2-GP and train from scratch: test RMSE: 0.31018739957 ( failed toan tap) - 50 iterations - pseudo inputs: 500

Different kernel functions do not contribute such a significant difference in result, at least from my experiment with quality estimation.

I used validdata.txt as training dataset

maternity kernel: 0.18460638297956458
RBF: 0.1837610677100123

neural network (baseline: 512-512-1): 0.190945125175 - both 500-50 iterations

neural network (baseline: 128-128-1): 0.187025368196 - 500 iterations


------- 26 sept 2017 ------

deep models seem do not help much for the task:

('Test RMSE:', 0.17422071640407702)

('Test RMSE 10:', 0.17892371045641844, 'dgp1 (sgp+adam)')

('Test RMSE 10:', 0.18064529673938259, 'dgp2')

('Test RMSE 10:', 0.18077692794954436, 'dgp3')

('Test RMSE 10:', 0.17599512661905775, 'dgp4')

('Test RMSE 10:', 0.17726577232048354, 'dgp5')


('Test RMSE:', 0.18207436990727555)

('Test RMSE 1:', 0.19461921459170606, 'dgp1 (sgp+adam)')

('Test RMSE 1:', 0.19486159103804393, 'dgp2')

('Test RMSE 1:', 0.19488038574956651, 'dgp3')

('Test RMSE 1:', 0.19482974425709446, 'dgp4')

('Test RMSE 1:', 0.19478574370120291, 'dgp5')


('Test RMSE 100:', 0.1713069200547977)

('Test RMSE 100:', 0.1750586335976152, 'dgp1 (sgp+adam)')

('Test RMSE 100:', 0.17379188690851397, 'dgp2')

('Test RMSE 100:', 0.17329473990766442, 'dgp3')

('Test RMSE 100:', 0.1744132242174119, 'dgp4') 

('Test RMSE 100:', 0.1740911970732055, 'dgp5')


('Test RMSE:', 0.17045979565507988)

('Test RMSE 500:', 0.17384460173684579, 'dgp1 (sgp+adam)')

('Test RMSE 500:', 0.17514294647823064, 'dgp2')


('Test RMSE 2000:', 0.17015620648697957)

('Test RMSE 2000:', 0.17013366883571904)

----------------------------------------------

ADAM with 20000 iterations

('Test RMSE 100 with 20000 iterations:', 0.17140373335968928, 'dgp1 (sgp+adam)')

('Test RMSE 100 with 20000 iterations:', 0.17282496214523227, 'dgp2')

('Test RMSE 100 with 20000 iterations:', 0.17364298469637351, 'dgp3')

('Test RMSE 100 with 20000 iterations:', 0.17219207311234672, 'dgp4') 

('Test RMSE 100 with 20000 iterations:', 0.17335594483617317, 'dgp5')


ADAM with 20000 iterations with noise

('Test RMSE 100 with 20000 iterations:', 0.1714037380048036, 'dgp1 (sgp+adam)')

('Test RMSE 100 with 20000 iterations:', 0.17184042425885582, 'dgp2')

('Test RMSE 100 with 20000 iterations:', 0.17368592962542548, 'dgp3')

('Test RMSE 100 with 20000 iterations:', 0.17389280085626924, 'dgp4')

('Test RMSE 100 with 20000 iterations:', 0.17421086259811297, 'dgp5')


--------

pseudo inputs: 100


('Test RMSE Matern52:', 0.17117225268902908)

('Test RMSE Matern32:', 0.17133212623686989)

('Test RMSE Matern12:', 0.17340097051503989)

('Test RMSE RBF 100:',  0.1713069200547977)

('Test RMSE Linear 100:',  0.2348395696106011)

Periodic - 0.18517560283550363

Polynomial (Degree = 12) - 0.24370107610661879

Polynomial (Degree = 9) - 0.17095645232169518

Polynomial (Degree = 7) - 0.17103013855422489

Polynomial (Degree = 6) - 0.17082216986546059

Polynomial (Degree = 5) - 0.17098419170801335

Polynomial (Degree = 4) - 0.17082629603808605

Polynomial (Degree = 3) - 0.17224442491626751

Polynomial (Degree = 2) - 0.17500370793167785

RBF * Poly (Degree 4): 0.17130055243657658

RBF + Poly (Degree 4): 0.1707790610698586



RBF * Periodic - 0.17134235343444676

RBF + Periodic - 0.17130237134630183

RBF + RBF - 0.17125735161416744

RBF * RBF - 0.17132585503045419

RBF * Linear - 0.17203751042779605

RBF + Linear 0.17122192777895939


RBF_17K_Plus: 0.17815446001528001

RBF_17K_dot: 0.17136592066520157

RBF * Matern52: 0.1712109978294574

FULLLLLLLLLLLLLLLLLL MODEL: ('Test RMSE:', 0.17010193413195204)

---------------
wider and deeper network:

512 - 512 - 512 - 512 - 512 - 1 :0.173468197523

512 - 512 - 512 - 512 - 1 :0.17365498

512 - 512 - 512 - 1 : 0.17288635691

512 - 512 -1: 0.172621675363

512 - 1:  0.171424115376


with deep learning kernel:

Baseline: 512 - 1

Our 500 iterations (var loss): 0.172780731643

Our 500 iterations (val MSE): 0.172792447103 


Our 500 iterations from scratch: 0.173187880796



wider:

32 - 1: 17.41

64 - 1: 17.30

128 - 1: 17.24

256 - 1: 17.14

512 - 1: 0.171475918103

1024 - 1: 0.17146686575

2048 - 1: 0.170902741883

16384 - 1: 0.17336833463

32768 - 1: 0.173209534763



-------
ARD = false

1- 0.18368356187250448

10 - 0.17856684327374944

100 - 0.17567893118902647)

500 - 0.17414626111351064

LSTM_Baseline.py batch_size128epochs250_size512: RMSE - 0.19548947809

parameters
model.kern.lengthscales transform:+ve prior:None

[  0.9060063 0.89626782   8.17821839   9.76969173   1.63464558
  17.87426877  11.02483475   9.12596714  54.91926957  12.26097875
55.00851352  53.58659854  64.0630409    5.46488718   6.82687469
   8.72636505  12.63110322]
   
   model.kern.variance transform:+ve prior:None
   [ 0.09477398]
   model.likelihood.variance transform:+ve prior:None
   [ 0.03135338]

Reduing number of features:

('Test RMSE 100 for Sparse Model ARD:', 0.17120104261380131)

detailed parameters for 500
model.kern.lengthscales transform:+ve prior:None
[  0.90805023   0.90086019   8.19019887   9.80181879   1.63185332
  18.31155775  10.48634048   8.53396368  11.65252109   5.47430326
   6.89946306   8.94510802  12.70521612]

model.kern.variance transform:+ve prior:None
[ 0.09850708]

model.likelihood.variance transform:+ve prior:None
[ 0.03130402] 

('Test RMSE 500 for Sparse Model ARD:', 0.17035261704836535)



sparse 4000

model.Z transform:(none) prior:None

[[ -5.09962713e-01  -4.17259912e-01   2.29502343e+00 ...,   3.44594448e-01
   -9.01652431e-01  -7.24113250e-01]
 [  1.60480066e+00   1.89723197e+00  -2.09495799e-02 ...,   2.66233713e-01
    6.80954118e+00   7.39025534e+00]
 [ -9.50055399e-01  -1.02997202e+00  -1.46552120e+00 ...,   5.64830601e-01
    4.77878103e-01   6.41474248e-01]
 ..., 
 [  2.34407047e+00   2.84040362e+00   6.84132671e-01 ...,   3.86728141e-03
    2.92653587e-01   1.30152379e+00]
 [  2.03683886e+00   1.95783313e+00  -4.23070787e-01 ...,  -7.95492383e-01
    6.77059462e-01   1.12320136e+00]
 [  3.10286001e-01   1.00509193e-01  -1.32244110e+00 ...,  -2.40456737e-01
   -8.19136228e-01  -6.87896974e-01]]
   
model.kern.lengthscales transform:+ve prior:None

[  0.7772421    0.76487953   6.69605738   4.34293202   1.30622618
   9.76556876   5.73416502   6.37394746  50.55233448   7.96823734
  48.76178375  55.21326922  63.88933522   4.68171942   6.0533537
   6.50611009   8.62859364]
   
model.kern.variance transform:+ve prior:None

[ 0.07099548]

model.likelihood.variance transform:+ve prior:None

[ 0.03065057]




--------------------------------------------- EN to DE 2017 -----------------------------------------

MLP - 128 - 250 iterations - ?

MLP - 256 - 250 iterations - ?

MLP - 512 - 250 iterations - 0.17462398879

MLP - 512 - 512 - 1 - 250 iterations - 0.174724199213

MLP - 512 - 512 - 512 - 1 - 250 iterations - 0.175116179608

MLP - 512 - 512 - 512 - 512 - 1 - 250 iterations - 0.174813963378

MLP - 1024 - 1 - 250 iterations: 0.174626321568

MLP - 2048 - 1 - 250 iterations: 0.173763737251

MLP - 8192 - 1 - 250 iterations: 0.174407684884

MLP - 16384 - 250 iterations: 0.175080060026

MLP - 32768 - 250 iterations: 0.174020661987


Sparse: 1 - 0.1776243390403813

Sparse: 10 - 0.17493865915681467

Sparse: 100 - 0.1739678798553802

Sparse: 500 - 0.17387987575208969

Sparse: 1000 - 0.17377791914385535

Sparse: 2000 - 0.1737636162882934

Sparse: 4000 - 0.17405597439575526

Sparse: 100 + deep learning kernel 0.173534931882

Sparse: 100 + deep learning kernel but from scratch 0.173541573892


Sparse: 100 + plus additive (1,2),  (3, 4), (16, 17): 0.1735something

Deep Gaussian Pocesses



('Test RMSE:', 0.1776243390403813)

('Test RMSE 1:', 0.17765471411164327, 'dgp1 (sgp+adam)')

('Test RMSE 1:', 0.1771819473212751, 'dgp2')

('Test RMSE 1:', 0.17698403453792905, 'dgp3')

('Test RMSE 1:', 0.17751157787918487, 'dgp4')

('Test RMSE 1:', 0.17655619411409237, 'dgp5')

('Test RMSE:', 0.17493865915681467)

('Test RMSE 10:', 0.1747557179044521, 'dgp1 (sgp+adam)')

('Test RMSE 10:', 0.17468796110874665, 'dgp2')

('Test RMSE 10:', 0.17463573428914381, 'dgp3')

('Test RMSE 10:', 0.1752323374203332, 'dgp4')

('Test RMSE 10:', 0.17543809904999294, 'dgp5')

('Test RMSE:', 0.1739678798553802)

('Test RMSE 100:', 0.17401374731410588, 'dgp1 (sgp+adam)')

('Test RMSE 100:', 0.17398644073617428, 'dgp2')

('Test RMSE 100:', 0.17402699153293996, 'dgp3')

('Test RMSE 100:', 0.17398118196450219, 'dgp4')

('Test RMSE 100:', 0.17409954441448988, 'dgp5')

--------------------------

Polynomial (Degree = 2) - 0.17418068420880428

Polynomial (Degree = 3) - 0.17411065149787217

Polynomial (Degree = 4) - 0.17412721980819795

Polynomial (Degree = 5) - 0.17412888655244219

Linear - 0.29463945272192954




--------------------------Domain Adaptation --------------------

train, valid: WMT 2016,
test: 2017

MLP -512-1: 0.208127499438

Sparse: 100: 0.20256593766647818

('Test RMSE 10:', 0.18961349236708896)

('Test RMSE 500:', 0.20234792620746286)

('Test RMSE 1:', 0.19473579999340698)

('Test RMSE 1000:', 0.20235635145549452)

MLP + GPs: 0.205873623406

MLP + GPs: 0.196797476351 - from scratch

Deep GPs

('Test RMSE 100:', 0.20127682571441879, 'dgp1 (sgp+adam)') 

('Test RMSE 100:', 0.20233598310362771, 'dgp2') 

('Test RMSE 100:', 0.20202339735969685, 'dgp3') 

('Test RMSE 100:', 0.20224315229721748, 'dgp4') 


Deep Network Adaptation: 512-1: 21.1433950323 - 21.2679994752


Deep Network Adaptation: 512-1: 21.1433950323 - 21.2679994752

Deep Network Adaptation: 512-512-1: 21.6995136291


----------------WMT 2015 - spanish --------------

('Test RMSE 100:', 0.18480514750284643) 

('Test RMSE 500:', 0.18445117171328618)

('Test RMSE 1000:', 0.18443653944955923)

('Time execution', 310.5293970108032)

('Test RMSE Adaptation:', 0.21906962472875163) 

('Test RMSE 2000:', 0.18443453682885252)

('Time execution', 1042.2259728908539) 

('Test RMSE:', 0.21913530794451477)

('Test RMSE 4000:', 0.18442362053829667)

('Time execution', 4092.636435985565) 

('Test RMSE Adaptation:', 0.21918294376478725)

('Test RMSE 10:', 0.18476864140316387) 


MLP: Test RMSE: 0.184830029689

Test Adaptation RMSE: 0.234363837146

Deep Network: 512-512-1 - 0.185001231168

Deep Network Adaptation: 512-512-1 - 0.205692874883

Deep Network: 512-512-512-1 - 0.184724029588

Deep Network Adaptation: 512-512-512-1 - 0.205057368213

Deep Network: 512-512-512-512-1 - 0.185154200703

Deep Network Adaptation: 512-512-512-512-1 - 0.204657531766

Deep Network: 512-512-512-512-512-1 - 0.184935222483

Deep Network Adaptation: 512-512-512-512-512-1 - 0.203610873839

Deep learning kernel:

Test RMSE: 0.18565758114

Test RMSE: 0.185417706629

-------------------------------------------------


*adaptation* with Spanish:

train, valid: english-spanish

test: english-german

MLP: Test RMSE: 0.234593462258

('Test RMSE 100:', 0.21255217587251546)

('Test RMSE 500:', 0.21918383628749555) 


---final run -...

('Test RMSE 1:', 0.18475441801078751, 'dgp1 (sgp+adam)')

('Test RMSE 1:', 0.21090247614407803, 'dgp1 (sgp+adam)') 


('Test RMSE 1:', 0.18485709211550266, 'dgp2')

('Test RMSE 1:', 0.21015237541079659, 'dgp2') 

('Test RMSE 1:', 0.1848285583683611, 'dgp3')

('Test RMSE 1:', 0.21096954957557043, 'dgp3')


('Test RMSE 1:', 0.18469350401901369, 'dgp4')

('Test RMSE 1:', 0.21063967703390407, 'dgp4')


('Test RMSE 1:', 0.1848695265518879, 'dgp5')

('Test RMSE 1:', 0.21030241345517139, 'dgp5')

-------------------------------


('Test RMSE 100:', 0.21123617187032856, 'dgp1 (sgp+adam)')

('Test RMSE 100:', 0.21088207981157298, 'dgp2')

('Test RMSE 100:', 0.21153051154871291, 'dgp3')

('Test RMSE 100:', 0.2115358703475414, 'dgp4')

('Test RMSE 100:', 0.21153440330067752, 'dgp5') 

Deep learning kernel:


Test RMSE: 0.199293826455 (from scratch)

Test RMSE: 0.207473532485

-------------------------------------

german to english adaptation to english to german

MLP: 0.249613619354

('Test RMSE for Sparse Model:', 0.21124365903641651)

Deep Network (normal): 512-512-1 - 0.172449183325

Deep Network Adaptation: 512-512-1 - 0.218127110626

Deep Network (normal): 512-512-512-1 - 0.173246190331

Deep Network Adaptation: 512-512-512-1 3layers - 0.228482146115

Deep Network (normal): 512-512-512-512-1 4 layers -0.173109383462

Deep Network Adaptation: 512-512-512-512-1 4layers - 0.22057771895 - 0.223584575035

Deep Network (normal): 512-512-512-512-1 5 layers -0.173039981369 - 0.174571540566 - 0.173457814642

Deep Network Adaptation: 512-512-512-512-1 5layers - 0.213737021682 - 0.218320387075 - 0.221980683165



Deep learning kernel:

Test RMSE: 0.224551159035

Test RMSE: 0.219251055049 - from scratch


Deep GPs - is running thunder 5.

('Test RMSE 100:', 0.21032053974189471, 'dgp1 (sgp+adam)')

('Test RMSE 100:', 0.22188420491117991, 'dgp2') 

('Test RMSE 100:', 0.22457417915245484, 'dgp3') 

-------------------

Random forest:

De-En: 0.171028683166

En-De: 0.174945230669

En-Spanish: 0.185372344454

En-Spanish: adaptation 0.195997505137

De-En and adaptation: 0.190948698564

WMT 2016 and adaptation: 0.189763546978


---------------------
Decision Tree

WMT 2015 - spanish - 0.187095134667

WMT 2015 - Spanish adaptation - 0.196092144759


WMT 2017 De-En: 0.184066768475

WMT 2017 En-DE: 0.178836140506

WMT 2017 DE-EN and adaptation: 0.214170907099

WMT 2016 Adaptation: 0.190259731417





---------------
GradientBoostingRegressor


WMT 2015 Spanish 0.184946787674

WMT 2015 Spanish Adaptation 0.200631416515

WMT 2016 adaptation: 0.246633948368

WMT 2017 EN-DE: 0.171446460982 

WMT 2017 DE-EN adaptation: 0.209265164046

WMT 2017 EN-De: 0.171446460982


----------------------
Relevance Vector Machine

WMT 2015 Spanish 0.184874995567

Spanish Adaptation 0.211766504431

WMT 2017 De-En: 0.175447055682

WMT 2017 DE-EN and adaptation: 0.219236638399?

WMT 2016 Adaptation: 0.198783201642

WMT 2017 En-De: 0.175761239079


------- SVR ---------

wmt 2017 de-en: 0.177934046869

RIDGE 2017 de-en: 0.174671584826

wmt 2017 de-en Adaptation to en-de: 0.206870889297

RIDGE 2017 de-en Adaptation to en-de: 0.286655597179

WMT 2015 spanish: 0.184126525764

WMT 2015 SVR RIDGE spanish: 0.184615864215

WMT 2015 Adaptation SVR DE: 0.226830752555

WMT 2015 Adaptation RIDGE DE: 0.211999637328


SVR WMT 2016 EN-DE adaptation: 0.202823967738

SVR RIDGE WMT 2016 EN-DE adaptation: 0.205540907673

SVR WMT 2017 EN-DE: 0.174648898981

SVR RIDGE WMT 2017 EN-DE: 0.174116620602


