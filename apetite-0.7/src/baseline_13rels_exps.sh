# base exps with 13 TML relations
python n_fold_experiment.py -o ../results/acq-13tmlrels ../data/aquaint_timeml_1.0/data/ 2> ../results/acq-13tmlrels.log 1> ../results/acq-13tmlrels.results
python n_fold_experiment.py -o ../results/tb-13tmlrels ../data/TimeBank1.1/docs/ 2> ../results/tb-13tmlrels.log 1> ../results/tb-13tmlrels.results
python n_fold_experiment.py -o ../results/otc-13tmlrels ../data/OTC/ 2> ../results/otc-13tmlrels.log 1> ../results/otc-13tmlrels.results
python n_fold_experiment.py -o ../results/acq-13tmlrels-sat -s ../data/aquaint_timeml_1.0/data/ 2> ../results/acq-13tmlrels-sat.log 1> ../results/otc-13tmlrels-sat.results
python n_fold_experiment.py -o ../results/tb-13tmlrels-sat -x ../data/TimeBank1.1/docs/ 2> ../results/tb-13tmlrels-beth.log 1> ../results/otc-13tmlrels-beth.results
python n_fold_experiment.py -o ../results/otc-13tmlrels-sat -x -s ../data/OTC/ 2> ../results/otc-13tmlrels-sat-beth.log 1> ../results/otc-13tmlrels-sat-beth.results






