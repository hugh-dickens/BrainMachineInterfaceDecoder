%%% Team Members:Hugh Dickens, Giorgio Martinelli, Rahel Ohlendorf, Michal
%%% Olak
%%% BMI Spring 2021 


function [modelParameters] = positionEstimatorTraining(training_data)
window_size = 20 ;
  
% Calc of velocity from position data. Change window size to reflect the length of interest
[x_vel_points, y_vel_points,spike_train_binned]= getvel(training_data, window_size) ;
[trials,angle]=size(training_data);
neurons=length(training_data(1,1).spikes(:,1));



%% Linear regression. Y = vel, X = binned 
for j=1:angle
x_lr =lsqminnorm(spike_train_binned(:,:,j), x_vel_points(:,j)); %getting coefficients
y_lr = lsqminnorm(spike_train_binned(:,:,j), y_vel_points(:,j));
model1(j).xlr=x_lr;
model2(j).ylr=y_lr;
end
modelParameters.model1 = model1;
modelParameters.model2 = model2;
 %% KNN Classifier
    % used to predict the reaching angle from the first 340ms

    spikes = [];
    direction = [];
    spike_angle = zeros(trials,neurons);

    for a = 1:angle
        for n = 1:neurons
            for t = 1:trials
                    spikesnr = sum(training_data(t,a).spikes(n,1:340));
                    spike_angle(t,n) = spikesnr;
            end
        end
        %spike_mean=mean(spike_angle,1);
        spikes = [spikes; spike_angle];
        angles(1:trials) = a;
        direction = [direction, angles];
        %direction=[1:1:8];
    end
      %size(spikes)
      %size(direction)
    knn = fitcknn(spikes,direction);
    %mdl=fitcdiscr(spikes,direction,'DiscrimType', 'linear','Delta',0.00028308,'Gamma',0.24285);%,'Prior','empirical');
    modelParameters.knn=knn;
    %modelParameters.mdl=mdl;
end

function [x_vel_points, y_vel_points, spike_train_binned]= getvel(TrainingData, window_size)
    
    [trial,angle] = size(TrainingData);
     x_vel_points = zeros(40000, 1,angle);
     y_vel_points = zeros(40000, 1,angle);
     neurons=length(TrainingData(1,1).spikes(:,1));
     spike_train_binned = zeros(40000,neurons,angle);
     counter2=1;
    for i= 1:trial
        for j=1:angle
            timesteps=length(TrainingData(i,j).handPos(1,:));
        for t = 320:window_size:timesteps-window_size
            valx(:,j)=TrainingData(i,j).handPos(1,window_size+t)-TrainingData(i,j).handPos(1,1+t);
            x_vel_points(counter2,j) = valx(:,j);
            valy(:,j)=TrainingData(i,j).handPos(2,window_size+t)-TrainingData(i,j).handPos(2,1+t);
            y_vel_points(counter2,j) = valy(:,j);
            val(1,:,j)= var(TrainingData(i,j).spikes(:,(1+t):(window_size+t)),1,2);
            spike_train_binned(counter2,:,j) = val(1,:,j);
            counter2 = counter2 +1;
        end
        end
    end
    x_vel_points(counter2-1:end,:)=[];
    y_vel_points(counter2-1:end,:)=[];
    spike_train_binned(counter2-1:end,:,:)=[];
end
