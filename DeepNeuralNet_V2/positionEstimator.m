%%% Team Members:Hugh Dickens, Giorgio Martinelli, Rahel Ohlendorf, Michal
%%% Olak
%%% BMI Spring 2021 

function [x, y] = positionEstimator(test_data, modelParameters)

    % Set the window and bin size same as for the training data!
    window_size = 300;
    bin_size = 30;
    
    % Collect the testing data!
    spike_train_test = getTestingData(test_data,window_size,bin_size);   

    % Now predict the changes in x and y using the neural network
    % already trained!
    prediction = predict(modelParameters,spike_train_test');
    dx=prediction(1,end);
    dy=prediction(2,end);
    
    if isempty(test_data.decodedHandPos)
        % If there is no decoded hand position yet, this is the start of
        % the testing and the next position will be calculated with respect
        % to the starting position
        
        x = test_data.startHandPos(1) + dx;
        y = test_data.startHandPos(2) + dy;
    else
        % If decoded hand position exists, the next position will be
        % calculated according to that
        
        x = test_data.decodedHandPos(1,end) + dx;
        y = test_data.decodedHandPos(2,end) + dy;
    end

end



function spike_train_test = getTestingData(TestData,window_size,bin_size)
% This function returns the testing inputs that will
% be used in the neural network architecture. 

    [neurons, time] = size(TestData(1,1).spikes); 

    number_bins = window_size/bin_size; 
    
    spike_train_test = zeros(1,neurons*number_bins);
    
    % collect the test data in the same way as training data
    for n = 1:number_bins
        bin = [time-n*bin_size:time-(n-1)*bin_size];
        spike_train_test(neurons*(n-1)+1:neurons*n) =  sum(TestData(1,1).spikes(:,bin),2);
    end
    
end

