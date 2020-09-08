function NetworkOutput = ApplyNeuralNetwork(Network,X,e)
K                                  = length(X(:,1));
Weights                            = Network.weights;
Topology                           = Network.Topology;
bias                               = Network.bias;
N_Layers                           = Network.N_Layers;
Nodes                              = zeros(K,sum(Topology(2:end)));
Connections                        = zeros(K,length(Nodes));
Prediction                         = zeros(K,Topology(end));
k=1;
    c_counter                      = 0;
    Nodes(1,1:Topology(1))         = X(1:Topology(1)); 
    for i=2:N_Layers
        if i==2
            n_counter = sum(Topology(1:i-1));
            for j=1:Topology(2)
                Temp               = 0;
                for ell=1:Topology(1)
                    c_counter      = c_counter + 1;
                    temp           = X(k,ell)*Weights(c_counter);
                    Temp           = Temp + temp;
                end
                Temp               = Temp+bias(c_counter)*1;
                n_counter          = n_counter + 1;
                Connections(n_counter) = Temp;
                Nodes(k,n_counter) = 1 / ( 1 + exp(-Temp) );
            end
            n0                     = 0;
        else
            n_counter = sum(Topology(1:i-1));
            for j=1:Topology(i)
                Temp               = 0;
                for ell=1:Topology(i-1)
                    c_counter      = c_counter + 1;
                    temp           = Nodes(k,n0+ell)*Weights(c_counter);
                    Temp           = Temp + temp;
                end
                n_counter          = n_counter+1;
                Temp               = Temp +bias(c_counter)*1;
                Connections(n_counter) = Temp;
                Nodes(k,n_counter) = 1 / ( 1 + exp(-Temp) );
            end
                n0                 = n0 + Topology(i-1);
        end
    end
Prediction(k,:)                    = Nodes(k,end);
NetworkOutput.Connections          = Connections;
NetworkOutput.Nodes                = Nodes;
NetworkOutput.Prediction           = Prediction;