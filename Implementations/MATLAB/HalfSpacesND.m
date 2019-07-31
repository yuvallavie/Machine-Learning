% ----------------------------------------------------------
                % Basic Linear Classifier
                % H = sign(<w,x> + b) - Half Spaces
                % L_D(h) - Px~D[h(x) != y]
                % L_S(h) - 0/1 Loss
                % Surrogate Loss - Hinge Loss = max(0,1-(f(x_i)*y)
                % f(x_i) = {-1,1}
                % The Realizeable Case
                % m(e,d) = m(e,d) = VC-Dimension + LOG(1/delta) / epsilon
% ----------------------------------------------------------

%% 
% Let (Omega,S,h,f) be the learning problem such that
% Omega = Real distribution
% S = I.I.D Samples
% h = hypothesis output by the learner
% f = labeling function for the realizeable case
% n = Dimensionality of the distribution
% m = size of sample
% epsilon = requested accuracy of the learner
% delta = maximum probability a learner will fail
n = 5;
eps = 0.1;
delta = 0.02;
vcDim = n+1;
m = ceil((vcDim + log(1/delta))/eps);
ds = 10^8; % Size of the real distribution, unknown in real situations
Omega = rand(ds,n);
realLabels = zeros(1,ds);
for i = 1:ds
    realLabels(i) = labeler(Omega(i,:),3);
end
%%
S = Omega(1:m,1:n); % Sample space sized m with dimension n
labels = realLabels(1:m);

%%
[Mdl,FitInfo] = fitclinear(S,labels);
prediction =  Mdl.predict(Omega);
risk = 0;
for i = 1:ds
    if(prediction(i) ~= realLabels(i))
        risk = risk + 1;
    end
end
risk = risk/ds;