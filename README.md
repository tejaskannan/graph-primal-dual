# Graph Primal-Dual: Self-Training Graph Neural Networks with Lagrangian Duality

This project uses graph neural networks to solve the uncapacitated minimum cost flow problem. The graph neural network is trained without labelled data. This technique uses separate models for the primal and dual problems, and these models are trained with the goal of minimizing the expected duality gap. This project is the subject of my Cambridge ACS dissertation.

# Usage
The command below specifies how to train new model. The model parameters are specified through a parameters JSON file. Examples of such files are in the folder ```model_params```.
```
python main.py --train --params <params-json-file>
```
To test a trained model, run the command below. This command will plot the flow graphs for a subset of test instances.
```
python main.py --test --model <path-to-trained-model>
```
Existing models can be found in the ```trained_models``` folder. Figures comparing the results from trained models are found in the ```figures``` folder.

Finally, to generate a new dataset, use the command below. This fetches the road graph (specified in the parameters) from [Open Street Map](https://www.openstreetmap.org/).
```
python main.py --generate --params <params-json-file>
```
Existing graphs and datasets are found in the folders ```graphs``` and ```datasets``` respectively.