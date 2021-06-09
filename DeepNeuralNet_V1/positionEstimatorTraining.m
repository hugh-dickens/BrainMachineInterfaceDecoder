%%% Team Members:Hugh Dickens, Giorgio Martinelli, Rahel Ohlendorf, Michal
%%% Olak
%%% BMI Spring 2021 


function [modelParameters] = positionEstimatorTraining(training_data)
  

window_size = 20 ;
  
% Calc of velocity from position data. Change window size to reflect the length of interest
[x_vel_points, y_vel_points,spike_train_binned]= getvel(training_data, window_size) ;
[trials,angle]=size(training_data);
neurons=length(training_data(1,1).spikes(:,1));

%% Neural Networks - deep learning. Y = vel, X = binned 
% useful sources: https://uk.mathworks.com/help/deeplearning/ref/trainingoptions.html
% https://uk.mathworks.com/help/deeplearning/ug/list-of-deep-learning-layers.html
% https://uk.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.elulayer.html


numFeatures = 98;
    % Create the neuron layers of Neural Net - this need to be optimised!
    numHiddenUnits_L1 = 100;
    numHiddenUnits_L2 = 250;
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
         'MaxEpochs',epochs, ...
         'MiniBatchSize',batch);

output = [x_vel_points(:),y_vel_points(:)];
spike_train_binned = reshape(spike_train_binned,[size(x_vel_points(:),1),98]);


% % find the model parameters - i.e. train it!
modelParameters = trainNetwork(spike_train_binned',output',layers,options);

 
end





function [x_vel_points, y_vel_points, spike_train_binned]= getvel(TrainingData, window_size)
    
    [trial,angle] = size(TrainingData);
     x_vel_points = zeros(40000, 1,angle);
     y_vel_points = zeros(40000, 1,angle);
     neurons=length(TrainingData(1,1).spikes(:,1));
     spike_train_binned = zeros(40000,angle,neurons);
     counter2=1;
    for i= 1:trial
        for j=1:angle
            timesteps=length(TrainingData(i,j).handPos(1,:));
        for t = 320:window_size:timesteps-window_size
            valx(:,j)=TrainingData(i,j).handPos(1,window_size+t)-TrainingData(i,j).handPos(1,1+t);
            x_vel_points(counter2,j) = valx(:,j);
            valy(:,j)=TrainingData(i,j).handPos(2,window_size+t)-TrainingData(i,j).handPos(2,1+t);
            y_vel_points(counter2,j) = valy(:,j);
            val(1,:,j)= sum(TrainingData(i,j).spikes(:,(1+t):(window_size+t)),2);
            spike_train_binned(counter2,j,:) = val(1,:,j);
            counter2 = counter2 +1;
        end
        end
    end
    x_vel_points(counter2-1:end,:)=[];
    y_vel_points(counter2-1:end,:)=[];
    spike_train_binned(counter2-1:end,:,:)=[];
end
