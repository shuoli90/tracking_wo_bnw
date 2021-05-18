#################### Import Section of the code #############################

try:
    import numpy as np	
    import torch
except Exception as e:
    print(e,"\nPlease Install the package")

#################### Import Section ends here ################################


class KalmanFilter(object): 
    """docstring for KalmanFilter"""
    
    def __init__(self, state, dt=1, stateVariance=100, measurementVariance=1,  method="Velocity"):
        super(KalmanFilter, self).__init__()
        self.method = method
        self.stateVariance = stateVariance
        self.measurementVariance = measurementVariance
        self.dt = dt
        self.predictedState = state.T
        self.state = state.T
        self.initModel()

    """init function to initialise the model"""
    def initModel(self): 

        self.A = torch.eye(self.predictedState.shape[0]).cuda()
        self.B = torch.eye(self.predictedState.shape[0]).cuda()
        self.H = torch.eye(self.predictedState.shape[0]).cuda()
        self.R = torch.diag(torch.tensor([2.27, 2.21, 5.47, 2.86])).cuda()
        self.Q = self.stateVariance * torch.eye(self.A.shape[0]).cuda()
        
        # self.erroCov = self.stateVariance * torch.eye(self.A.shape[0]).cuda()
        self.erroCov = 10.0 * torch.eye(self.A.shape[0]).cuda()
        self.predictedErrorCov = 10.0 * torch.eye(self.A.shape[0]).cuda()


    """Predict function which predicst next state based on previous state"""
    def predict(self, vel):
        self.predictedState = torch.matmul(self.A, self.state) + torch.matmul(self.B, vel.T)
        self.predictedErrorCov = torch.matmul(torch.matmul(self.A, self.erroCov), self.A.T) + self.Q
        # breakpoint()

    """Correct function which correct the states based on measurements"""
    def correct(self, currentMeasurement):
        self.kalmanGain = self.predictedErrorCov@self.H.T@torch.pinverse(self.H@self.predictedErrorCov@self.H.T+self.R)
        self.state = self.predictedState + self.kalmanGain@(currentMeasurement.T - (self.H@self.predictedState))
        self.erroCov = (torch.eye(self.erroCov.shape[0]).cuda() - self.kalmanGain @ self.H) @ self.predictedErrorCov
        # breakpoint()

