%%% Team Members:Hugh Dickens, Giorgio Martinelli, Rahel Ohlendorf, Michal
%%% Olak
%%% BMI Spring 2021 (Update 17th March 2015)

function [x, y, newModelParameters] = positionEstimator(past_current_trial, modelParameters)
window_size = 20; %window size used for extraction of position from veloity.
%change according to the window size used for binning

spike_train_test= past_current_trial.spikes;%Loading the spike train given for testing


[neurons,length] = size(spike_train_test); %let's check the dimensions of what we work with
%the training data used bins. Therefore we need to bin the test data too
spike_train_binned_test = zeros(100000, neurons); %creating placeholder for the binned spike train.
spike_angle = zeros(neurons,1);

if length <= 320
    % find number of spikes
    for n = 1:neurons
        spikesnr = sum(spike_train_test(n,1:320));
        spike_angle(n) = spikesnr;
    end

% find most predicted direcion and set it as hand direction/reaching angle
    j = predict(modelParameters.lda,spike_angle');
    
 else
     % use value that was found prevously
     j = modelParameters.direction;
end
    counter = 1;%counter is for accesing a bin of interest
    for t = 300:window_size:length-window_size %looping over data from a single neuron.
        
        spike_train_binned_test(counter,:) = var(spike_train_test(:,(1+t):(window_size+t)),1,2);
        %this is where we calculate the value of bin.
        %binning is basically looking at spike data using
        %sliding,non-overlaping window. We calculate how many spikes there
        %are in a given window and take mean of that
        counter=counter+1;
    end
spike_train_binned_test(counter-1:end,:) = [];
%importing the linear regression model parameters


x_lr = modelParameters.model1(j).xlr;
y_lr = modelParameters.model2(j).ylr;
x_knn = modelParameters.model3(j).xknn;
y_knn = modelParameters.model4(j).yknn;

%let's predict the velocities!
x_vels_knn = predict(x_knn,spike_train_binned_test); 
y_vels_knn = predict(y_knn,spike_train_binned_test);
x_vels_lr=spike_train_binned_test*x_lr;
y_vels_lr=spike_train_binned_test*y_lr;
x_vels = x_vels_lr*0.75 + x_vels_knn*0.25;
y_vels = y_vels_lr*0.75 + y_vels_knn*0.25;
dx=sum(x_vels,1);
dy=sum(y_vels,1);
%okay, we've got velocities. Now it's time to extract positions from them
  x = past_current_trial.startHandPos(1) + dx;
  y = past_current_trial.startHandPos(2) + dy;
  newModelParameters.model1 = modelParameters.model1;
  newModelParameters.model2 = modelParameters.model2;
  newModelParameters.model3 = modelParameters.model3;
  newModelParameters.model4 = modelParameters.model4;
  newModelParameters.lda=modelParameters.lda;
  newModelParameters.direction = j;
%   if length <= 320
% 
%     x = past_current_trial.startHandPos(1);
% 
%     y = past_current_trial.startHandPos(2);
% 
% elseif length <= 500
% 
%     % position = previous position + velocity * 20ms
% 
%     x = past_current_trial.decodedHandPos(1,end) +dx;
% 
%     y = past_current_trial.decodedHandPos(2,end) + dy;
% 
% else
% 
%     x = past_current_trial.decodedHandPos(1,end);
% 
%     y = past_current_trial.decodedHandPos(2,end);
% 
% end



end
