
# EVOLUTION EVALUATIONS
python3 main_evolve.py aggregation -evaluate data/aggregation/evolution_aggregation_t200_1.pkl -reruns 100 -nmin 30 -nmax 30
python3 main_evolve.py pfsm_exploration -evaluate data/pfsm_exploration/evolution_pfsm_exploration_t200_1.pkl -reruns 100 -nmin 30 -nmax 30
python3 main_evolve.py pfsm_exploration_mod -evaluate data/pfsm_exploration_mod/evolution_pfsm_exploration_mod_t200_1.pkl -reruns 100 -nmin 30 -nmax 30
python3 main_evolve.py forage -evaluate data/forage/evolution_forage_t200_1.pkl -reruns 100 -nmin 20 -nmax 20