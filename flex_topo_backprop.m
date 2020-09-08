clc; clear all; format long;
nx                = 1;
ny                = 1;
K                 = 1000;
X                 = randn(K,nx);         
Y                 = sin(X);% + cos(X(:,2));   
Topology          = [nx 5 1 2 ny];
N_nodes           = 0;
N_Weights         = 0;
for i=1:(length(Topology) - 1)
    N_Weights     = N_Weights + Topology(i) * Topology(i+1);
    N_nodes       = N_nodes + Topology(i);
end
Network.weights   = ones(1,N_Weights);
Network.bias      = ones(1,N_Weights);
Network.N_Layers  = length(Topology);
Network.N_nodes   = N_nodes; 
Network.Topology  = Topology;
Network.N_Weights = N_Weights;
K                 = length(X); 
Train.inputs      = X;
TrainOutput       = TrainNeuralNetwork(Network,X,Y,N_Weights,Train);
Network.weights   = TrainOutput.wts;                                                          %update the trained weights
NetworkOutput     = ApplyNeuralNetwork(Network,X(1,:),K)                                        % Final Outputs
