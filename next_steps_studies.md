Step1:
grid/bayesian general search to find optimal default values?
OVER   1e-4 x (1,4,8,16) (rank) x (1,2,4,8,12) (layers)
-> discuss results 

Step2: 
After discussion and decision of default values: (each of us picks one parameter space and variies each option x4 times for seed varation)
first only scan one at a time, take for the other settings that are good as default, extremes also included!
(grid search over:) 
-rank :  1->(2)->4->8->16 -> (highest) prediction accuracy, increase -> saturation -> overfitting reduction 
(-learning rate : (0.1),0.01,0.001,0.0001, 0.000001 (1e-5))
-layers : last layer(layer12), last2 (layer, last4, last8, full(12)?, (also only lower layers?) -> look   -> since a tokenizer last layer prof agrees
(12-12),(12-10),(12-8),(12-4), (12-2), (12-1), (1), (4-1)

-conduct each experiment 4times for seed variance


Potential Hypothesis for us (still in progress, currently just draft)
For Layers:
-> Hypothese1: prediction accuracy is already increasing when only the last layer is updated
-> Hypothese2: prediction accuracy is still increasing with updating more layers starting from last layer, but saturating
-> Hypothese3: we dont see a similar increaserelativly in prediction accuracy for only fineuting the lower layers and adding more (e.g. 4-1) does not help
as much as from higher layers?
