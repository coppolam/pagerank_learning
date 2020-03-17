# Build build swarmulator
cd ../swarmulator
make clean
make -j

# Set a time limit for the evolution
cd conf
xmlstarlet edit -L -u "parameters/time_limit" -v '2000' parameters.xml
cd ../

# Run python code
cd ../pagerank_learning
python3 main_graphless.py
