function  TrainOutput = TrainNeuralNetwork(Network,X,Y,N_Weights,Train)
nit                              = 1;
NetworkOutput                    = ApplyNeuralNetwork(Network,X(1,:),nit);        % To find the first prediction value for training 
Weights                          = Network.weights;
N_Layers                         = Network.N_Layers;
Nodes                            = NetworkOutput.Nodes;
Connections                      = NetworkOutput.Connections;
Topology                         = Network.Topology;
error                            = 1;                                        % Set 1 to enter into the while_loop  
K                                = 100;
Gradient                         = zeros(1,length(Nodes));
while error > 0.1
    NetworkOutput                    = ApplyNeuralNetwork(Network,X(nit),nit);
    Prediction                   = NetworkOutput.Prediction;
    c_counter                    = N_Weights;
    n_counter                    = length(Nodes);
    LR                           = 0.01;
    for i=1:Topology(N_Layers)
        Iteration = nit
        Gradient(n_counter)      = - ( Y(nit) - Prediction ) * Prediction * ( 1 - Prediction);            
        c_counter                = c_counter - 1;
        n_counter                = n_counter - 1;
    end
    k_counter                    = length(Nodes)-Topology(end);
    n_counter                    = length(Nodes);
    for k = (N_Layers-1):-1:2
        for j = 1:Topology(k)
            sumTemp                  = 0;                
                for i = 1:Topology(k+1)
                    sumTemp          = sumTemp +Weights(c_counter)*Gradient(n_counter);
                    c_counter        = c_counter-Topology(k);
                    n_counter        = n_counter-1;
                end
                Gradient(k_counter)  = Nodes(k_counter)*(1-Nodes(k_counter))*sumTemp; 
                n_counter            = n_counter+Topology(k+1);     
                c_counter            = c_counter+Topology(k) * Topology(k+1) -1 ;
                k_counter            = k_counter-1;
        end  
    end
        %......... updates weights ...........
        k_counter                    = 1;
        c_counter                    = 1;
        m_counter                    = 1;
        for i = 1: (N_Layers-1)
            for j = 1: Topology(i+1)
                if (Topology(i) == 1 && i ==1)
                    m_counter              = 1;
                end
                for k = 1: Topology(i)
                    Weights(c_counter)     = Weights(c_counter) - LR*Gradient(k_counter)*Nodes(m_counter);
                    c_counter              = c_counter + 1;
                    if Topology(i) > 1
                        m_counter          = m_counter +1;
                    end
                end
                k_counter                  = k_counter +1;
                m_counter                  = sum(Topology(1:i-1))+1;
            end

        end  
        %......... updates weights ...........  
    error                       = 0.5*(Y(nit)-NetworkOutput.Prediction)^2
    TrainOutput.wts             = Weights;
    Network.weights             = Weights;          
    NetworkOutput               = ApplyNeuralNetwork(Network,X(1,:));               % for calculating new Prediction value       
    TrainOutput.Gradient        = Gradient;
    nit                         = nit+1;
    if nit > K
        nit = 1;
    end
end
