#################### Import Section of the code #############################

try:
    import numpy as np	
    import torch
except Exception as e:
    print(e,"\nPlease Install the package")

#################### Import Section ends here ################################


class KalmanFilter(object): 
    """docstring for KalmanFilter"""
    
    def __init__(self, state, dt=1, stateVariance=10, measurementVariance=1,  method="Velocity"):
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

        # self.A = np.matrix(np.identity(self.init_state.shape[0]))
        self.A = torch.eye(self.predictedState.shape[0])
        # self.B = np.matrix(np.identity(self.init_state.shape[0]))
        self.B = torch.eye(self.predictedState.shape[0])
        # self.H = np.matrix(np.identity(self.init_state.shape[0]))
        self.H = torch.eye(self.predictedState.shape[0])
        # self.P = np.matrix(self.stateVariance*np.identity(self.A.shape[0]))
        # self.R = np.matrix(self.measurementVariance*np.identity(self.H.shape[0]))
        # self.R = self.measurementVariance * torch.eye(self.predictedState.shape[0])
        self.R = torch.diag(torch.tensor([11.25, 5.25, 13.70, 11.01]))
        # self.Q = np.matrix(self.stateVariance*np.identity(self.A.shape[0]))
        self.Q = self.stateVariance * torch.eye(self.A.shape[0])
        
        # self.erroCov = np.matrix(self.stateVariance*np.identity(self.A.shape[0]))
        self.erroCov = self.stateVariance * torch.eye(self.A.shape[0])
        # self.predictedState = self.init_state
        self.predictedErrorCov = self.erroCov


    """Predict function which predicst next state based on previous state"""
    def predict(self, vel):
        self.predictedState = torch.matmul(self.A, self.state) + torch.matmul(self.B, vel.T)
        self.predictedErrorCov = torch.matmul(torch.matmul(self.A, self.erroCov), self.A.T) + self.Q
        temp = np.asarray(self.predictedState)
        return temp[0], temp[2]

    """Correct function which correct the states based on measurements"""
    def correct(self, currentMeasurement):
        self.kalmanGain = self.predictedErrorCov@self.H.T@np.linalg.pinv(self.H@self.predictedErrorCov@self.H.T+self.R)
        self.state = self.predictedState + self.kalmanGain@(currentMeasurement.T - (self.H@self.predictedState))
        self.erroCov = (torch.eye(self.erroCov.shape[0]) - self.kalmanGain*self.H) @ self.predictedErrorCov
        # breakpoint()

