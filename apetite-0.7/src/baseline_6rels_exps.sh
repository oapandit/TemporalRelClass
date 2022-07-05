# make sure exec's are in path
resource

# base exps with 6 TML relations
python n_fold_experiment.py -c -o ../results/acq-6tmlrels ../data/aquaint_timeml_1.0/data/ 2> ../results/acq-6tmlrels.log 1> ../results/acq-6tmlrels.results
python n_fold_experiment.py -c -o ../results/tb-6tmlrels ../data/TimeBank1.1/docs/ 2> ../results/tb-6tmlrels.log 1> ../results/tb-6tmlrels.results
python n_fold_experiment.py -c -o ../results/otc-6tmlrels ../data/OTC/ 2> ../results/otc-6tmlrels.log 1> ../results/otc-6tmlrels.results
python n_fold_experiment.py -c -o ../results/acq-6tmlrels-sat -s ../data/aquaint_timeml_1.0/data/ 2> ../results/acq-6tmlrels-sat.log 1> ../results/otc-6tmlrels-sat.results
python n_fold_experiment.py -c -o ../results/tb-6tmlrels-sat -x ../data/TimeBank1.1/docs/ 2> ../results/tb-6tmlrels-beth.log 1> ../results/otc-6tmlrels-beth.results
python n_fold_experiment.py -c -o ../results/otc-6tmlrels-sat -x -s ../data/OTC/ 2> ../results/otc-6tmlrels-sat-beth.log 1> ../results/otc-6tmlrels-sat-beth.results






