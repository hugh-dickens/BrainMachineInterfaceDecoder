%%% Team Members:Hugh Dickens, Giorgio Martinelli, Rahel Ohlendorf, Michal
%%% Olak
%%% BMI Spring 2021 


function [modelParameters] = positionEstimatorTraining(training_data)
bin_size = 30 ;
window_size = 300; 
  
% Calc of velocity from position data. Change window size to reflect the length of interest
[output,spike_train_binned]= getTraining(training_data, window_size, bin_size);

%% Neural Networks - deep learning. Y = vel, X = binned 
% useful sources: https://uk.mathworks.com/help/deeplearning/ref/trainingoptions.html
% https://uk.mathworks.com/help/deeplearning/ug/list-of-deep-learning-layers.html
% https://uk.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.elulayer.html

    % inputs i.e. 98 neurons timesed by the number of bins used for each spiking neuron from monkeys!
    numFeatures = 98*window_size/bin_size;
    % Create the neuron layers of Neural Net - this need to be optimised!
    numHiddenUnits_L1 = 100;
    numHiddenUnits_L2 = 150;
    numHiddenUnits_L3 = 100;
    numHiddenUnits_L4 = 50;
    % Self explanatory
    LearningRate = 0.01;
    batch = 30; % Minibatch size
    epochs = 60; % Amount of epochs
    % % We have 2 outputs, x and y velocities/ changes
    numResponses = 2;

    layers = [
        sequenceInputLayer(numFeatures)

        fullyConnectedLayer(numHiddenUnits_L1)
        tanhLayer

        fullyConnectedLayer(numHiddenUnits_L2)
        tanhLayer
        
        dropoutLayer

        fullyConnectedLayer(numHiddenUnits_L3)
        tanhLayer       
        
        fullyConnectedLayer(numHiddenUnits_L4)
        tanhLayer       
        
        fullyConnectedLayer(numResponses)
        regressionLayer];

    options = trainingOptions('adam', ...
         'InitialLearnRate',LearningRate, ...
         'ExecutionEnvironment','cpu', ...
         'MaxEpochs',epochs, ...
         'MiniBatchSize',batch)
         

    % % find the model parameters - i.e. train it!
    modelParameters = trainNetwork(spike_train_binned',output',layers,options);

end

function [output, spike_train_binned]= getTraining(TrainingData, window_size, bin_size)
  
     [trial,angle] = size(TrainingData);
     output = zeros(400000, 2);
     number_bins = window_size/bin_size;
     neurons=length(TrainingData(1,1).spikes(:,1));
     spike_train_binned = zeros(50000,number_bins*neurons);
     val = zeros(1,neurons*number_bins); 
    
    counter=1;
    for i= 1:trial
        for j=1:angle
            timesteps=length(TrainingData(i,j).handPos(1,:));
            for t = 320:bin_size:timesteps
                % this assigns change in x to first column
                output(counter,1)=TrainingData(i,j).handPos(1,t)-TrainingData(i,j).handPos(1,t-20);
                % this assigns change in y to second column
                output(counter,2)=TrainingData(i,j).handPos(2,t)-TrainingData(i,j).handPos(2,t-20);
                
                for n=1:number_bins
                    % bin is used to sum over the bins, this looks
                    % backwards in time
                    bin = [t-n*bin_size:t-(n-1)*bin_size];
                    val(neurons*(n-1)+1:neurons*n)= sum(TrainingData(i,j).spikes(:,bin),2);

                end
                % Assign the temporary value variable to the spike training
                % binned dataset
                spike_train_binned(counter,:) = val;
                counter = counter +1;
            end
        end
    
        output = output(1:counter-1,:);
        spike_train_binned = spike_train_binned(1:counter-1,:);
    end
end
