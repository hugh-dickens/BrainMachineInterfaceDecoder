%%% Team Members:Hugh Dickens, Giorgio Martinelli, Rahel Ohlendorf, Michal
%%% Olak
%%% BMI Spring 2021 (Update 17th March 2015)

function [x, y] = positionEstimator(past_current_trial, modelParameters)
window_size = 20; %window size used for extraction of position from veloity.
%change according to the window size used for binning

spike_train_test= past_current_trial.spikes;%Loading the spike train given for testing


[neurons,length] = size(spike_train_test); %let's check the dimensions of what we work with
%the training data used bins. Therefore we need to bin the test data too
spike_train_binned_test = zeros(length/window_size, neurons); %creating placeholder for the binned spike train. length/window_size gives us the number of bins
%spike_train_binned_test = zeros(length/window_size, 392);

    counter = 1;%counter is for accesing a bin of interest
    for t = 100:window_size:length-window_size %looping over data from a single neuron.
        %for n=80/20
                %bin = [t-n*window_size:t-(n-1)*window_size];
        spike_train_binned_test(counter,:) = sum(spike_train_test(:,(1+t):(window_size+t)),2);
        %spike_train_binned_test(counter,:) = sum(spike_train_test(:,bin),2);
        %spike_train_binned_test(counter,neurons*(n-1)+1:neurons*n) = sum(spike_train_test(:,bin),2);
        %spike_train_binned_test(counter,:) = sum(spike_train_test(:,bin),2);
        %this is where we calculate the value of bin.
        %binning is basically looking at spike data using
        %sliding,non-overlaping window. We calculate how many spikes there
        %are in a given window and take mean of that
        %counter=counter+1;
        %end
        counter = counter +1;
    end

%importing the linear regression model parameters
x_lr = modelParameters.model1;
y_lr = modelParameters.model2;

%let's predict the velocities!
%[x_vels_lr, ~] = predict(x_lr,spike_train_binned_test); 
%[y_vels_lr, ~] = predict(y_lr,spike_train_binned_test);
x_vels_lr=spike_train_binned_test*x_lr;
y_vels_lr=spike_train_binned_test*y_lr;
dx=sum(x_vels_lr,1);
dy=sum(y_vels_lr,1);
%okay, we've got velocities. Now it's time to extract positions from them
  x = past_current_trial.startHandPos(1) + dx;
  y = past_current_trial.startHandPos(2) + dy;
 if length <= 530

    x = past_current_trial.startHandPos(1) + dx;

    y = past_current_trial.startHandPos(2) + dy;

else

    x = past_current_trial.decodedHandPos(1,end);

    y = past_current_trial.decodedHandPos(2,end);
 end
  
end
