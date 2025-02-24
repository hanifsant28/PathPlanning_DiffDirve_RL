import numpy as np


class DifDriveKinematic:
    def __init__(self,
                 wheel_radius = 0.08,
                 wheel_base = 0.15,
                 dt = 0.001):
        
        """
        Initialize the differential drive robot with given physical parameters.
        
        Parameters:
            wheel_radius (float): Radius of the wheels (m).
            wheel_base (float): Distance between the wheels (m).
            dt (float): time increment (s)
        """

        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base

        self.dx = 0
        self.dy = 0
        self.dTheta = 0

        self.theta = 0
        self.x = 0
        self.y = 0

        self.dt = dt


    def computeKinematic(self,wR,wL):
        
            # Compute angular velocity
        self.dTheta = (self.wheel_radius / self.wheel_base) * (wR - wL)

        # Compute forward velocity
        v = (self.wheel_radius / 2) * (wR + wL)

        # Compute linear velocity in global frame
        self.dx = v * np.sin(-self.theta)
        self.dy = v * np.cos(-self.theta)

        # Update position and orientation
        self.x += self.dx * self.dt
        self.y += self.dy * self.dt
        self.theta += self.dTheta * self.dt  # Apply dt here for correct integration

    
    def get_state(self):
        """
        Retrieves the current state of the mobile robot.

        Returns:
            np.ndarray: A NumPy array containing:
                - x (float): X-coordinate of the robot
                - y (float): Y-coordinate of the robot
                - theta (float): Orientation (angle in radians)
                - dx (float): Velocity along the X-axis
                - dy (float): Velocity along the Y-axis
                - dTheta (float): Angular velocity (rate of change of theta)
        
        Example:
            If self.x = 1.0, self.y = 2.0, self.theta = 0.5,
            self.dx = 0.1, self.dy = 0.2, self.dTheta = 0.05, 
            then the function returns:
            
            np.array([1.0, 2.0, 0.5, 0.1, 0.2, 0.05])
        """

        return np.array([self.x,self.y,self.theta,self.dx,self.dy,self.dTheta])


    def reset(self):
        """
        Function to reset state to 0
        """
        
        self.dx = 0
        self.dy = 0
        self.dTheta = 0

        self.theta = 0
        self.x = 0
        self.y = 0

        
