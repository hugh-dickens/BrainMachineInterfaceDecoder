%%% Team Members:Hugh Dickens, Giorgio Martinelli, Rahel Ohlendorf, Michal
%%% Olak
%%% BMI Spring 2021 (Update 17th March 2015)

function [x, y] = positionEstimator(past_current_trial, modelParameters)
window_size = 20; %window size used for extraction of position from veloity.
%change according to the window size used for binning

spike_train_test= past_current_trial.spikes;%Loading the spike train given for teting


[neurons,length] = size(spike_train_test); %let's check the dimensions of what we work with
%the training data used bins. Therefore we need to bin the test data too
spike_train_binned_test = zeros(length/window_size, neurons); %creating placeholder for the binned spike train. length/window_size gives us the number of bins

counter = 1;%counter is for accesing a bin of interest

for i = 300:window_size:length-window_size %looping over data from a single neuron.
    spike_train_binned_test(counter,:) = sum(spike_train_test(:,(1+i):(window_size+i)),2); %this is where we calculate the value of bin.
    %binning is basically looking at spike data using
    %sliding,non-overlaping window. We calculate how many spikes there
    %are in a given window and take mean of that
    counter = counter +1;
end


%let's predict the velocities of the Neural Network!
% useful resources: https://uk.mathworks.com/help/deeplearning/ref/seriesnetwork.predict.html

% % two nets - one on x and one on y 
output  = predict(modelParameters,spike_train_binned_test');

dx=sum(output(1,:));
dy=sum(output(2,:));

 if length <= 530

    x = past_current_trial.startHandPos(1) + dx;

    y = past_current_trial.startHandPos(2) + dy;

else

    x = past_current_trial.decodedHandPos(1,end);

    y = past_current_trial.decodedHandPos(2,end);
 end
% 
% %okay, we've got changes in position. Now it's time to extract position of the hand
% % in x and y coordinates
% x = past_current_trial.startHandPos(1) + dx;
% y = past_current_trial.startHandPos(2) + dy;
%   
end
