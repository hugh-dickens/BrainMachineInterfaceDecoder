%%% Team Members:Hugh Dickens, Giorgio Martinelli, Rahel Ohlendorf, Michal
%%% Olak
%%% BMI Spring 2021 


function [modelParameters] = positionEstimatorTraining(training_data)
window_size = 20 ;
  
% Calc of velocity from position data. Change window size to reflect the length of interest
[x_vel_points, y_vel_points,spike_train_binned]= getvel(training_data, window_size) ;
% Binning all spike trains. Window size is taken from velocity
% size(spike_train_binned)
% size(x_vel_points)
% size(y_vel_points)
%% Reshape matrices. This window has to be run once and not more due to the way 'permute' function works
%spike_train_binned=permute(spike_train_binned, [2,3,4,1]);
%x_vel_points = reshape(x_vel_points, [size(x_vel_points,1)*size(x_vel_points,2)*size(x_vel_points,3), 1]);
%y_vel_points = reshape(y_vel_points, [size(y_vel_points,1)*size(y_vel_points,2)*size(y_vel_points,3), 1]);
%spike_train_binned = reshape(spike_train_binned, [size(spike_train_binned,1)*size(spike_train_binned,2)*size(spike_train_binned,3),size(spike_train_binned,4)]);


%% Linear regression. Y = vel, X = binned 

x_lr = lsqminnorm(spike_train_binned, x_vel_points); %getting coefficients
y_lr = lsqminnorm(spike_train_binned, y_vel_points);
modelParameters.model1 = x_lr ;
modelParameters.model2 = y_lr;

end

function [x_vel_points, y_vel_points, spike_train_binned]= getvel(TrainingData, window_size)
    [trial,angle] = size(TrainingData);
     x_vel_points = zeros(4000000, 1);
     y_vel_points = zeros(40000000, 1);
     neurons=length(TrainingData(1,1).spikes(:,1));
    spike_train_binned = zeros(4000000,neurons);
    %spike_train_binned = zeros(40000,392);
    counter2=1;
    for i= 1:trial
        for j=1:angle
            timesteps=length(TrainingData(i,j).handPos(1,:));
        for t = 100:window_size:timesteps-window_size
            valx=TrainingData(i,j).handPos(1,window_size+t)-TrainingData(i,j).handPos(1,1+t);
            %valx=TrainingData(i,j).handPos(1,t)-TrainingData(i,j).handPos(1,t-20);
            x_vel_points(counter2) = valx;
            valy=TrainingData(i,j).handPos(2,window_size+t)-TrainingData(i,j).handPos(2,1+t);
            %valy=TrainingData(i,j).handPos(2,t)-TrainingData(i,j).handPos(2,t-20);
            y_vel_points(counter2) = valy;
            %for n=80/20
                %bin = [t-n*window_size:t-(n-1)*window_size];
            val(1,:)= sum(TrainingData(i,j).spikes(:,(1+t):(window_size+t)),2);
            %val(neurons*(n-1)+1:neurons*n)= sum(TrainingData(i,j).spikes(:,bin),2);
            %val(1,:)= sum(TrainingData(i,j).spikes(:,bin),2);
            spike_train_binned(counter2,:) = val;
            %counter2 = counter2 +1;
            %end
            counter2 = counter2 +1;
        end
        end
    end
    x_vel_points(counter2-1:end)=[];
    y_vel_points(counter2-1:end)=[];
    spike_train_binned(counter2-1:end,:)=[];
end
