import numpy as np
from kinematic import DifDriveKinematic
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon



class environment:
    def __init__(self,
                 arena_width,
                 arena_length,
                 robot_width,
                 robot_length,
                 robot_pos,
                 barrier_pos,
                 barrier_radius,
                 target_pos,
                 num_lidarRay,
                 num_lidarBatch,
                 lidar_range,
                 max_speed,
                 max_time,
                 dt,
                 step_num):
        
        """
        Initialize the environment for the robot.
        
        Parameters:
            - arena_width (float): width of the arena
            - arena_length (float): length of the arena
            - robot_width (float): width of the robot
            - robot_length (float): length of the robot
            - robot_pos (list of float): initial position of the robot [x,y]
            - barrier_pos (2d list of float): position of the barrier [[x1,y1],[x2,y2],...]
            - barrier_radius (list of float): radius of the barrier[rad1,rad2,...]
            - target_pos (float): position of the target
            - num_lidarRay (int): number of LIDAR rays
            - num_lidarBatch (int): number of LIDAR batch
            - lidar_range (float): range of the LIDAR
            - max_speed (float): maximum speed of the robot
            - max_time (float): maximum time for the robot to reach the target
            - dt (float): time step
            - step_num (int): number of step for each action
            
        """

        # Colors
        self.WHITE = (255, 255, 255)
        self.GRAY = (50, 50, 50)
        self.YELLOW = (255, 255, 0)
        self.RED = (200, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 200, 0)
        self.BLACK = (0, 0, 0)
        self.ORANGE = (255, 165, 0)


        self.arena_width = arena_width
        self.arena_length = arena_length
        
        self.step_num = step_num

        self.max_speed = max_speed
        self.dt = dt
        self.max_time = max_time
        self.time_passed = 0

        #size calculation for robot
        self.robot_width = robot_width
        self.robot_length = robot_length

        #calculation of wheel size
        self.wheel_width = self.robot_width * 0.2
        self.wheel_length = self.robot_length * 0.333334

        #intialize robot object
        wheel_radius = self.wheel_length/2
        self.my_robot = DifDriveKinematic(wheel_radius,self.robot_width,self.dt)

        #positon and state of robot based on the reading from the robot model
        self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read = self.my_robot.get_state()

        #position of the robot in real  based on our coordinate taken from the middle of the robot
        self.start_x = robot_pos[0]
        self.start_y = robot_pos[1]

        self.x_real = robot_pos[0]
        self.y_real = robot_pos[1]

        #intialize position of the target
        self.target_pos = target_pos

        #number of lidar rays place to save lidar reading
        self.lidar_range = lidar_range
        self.lidar_batch_num = num_lidarBatch
        self.num_lidarRays = num_lidarRay
        self.lidar_dis_read = [] #real world reading
        self.lidar_dis_min = [] #minimum reading of each batch

        #intialize barrie position
        self.barrier_pos = barrier_pos
        self.barrier_radius = barrier_radius

        #buffer for replay in pygame
        self.robot_pos_buffer = []
        self.wheelL_pos_buffer = []
        self.wheelR_pos_buffer = []
        self.lidar_pos_pair_buffer = []
        self.lidar_dis_buffer = []

        #flag to indicate whether the episode finsih or not
        self.episode_end = False

        #flag to indicate whether the robot or wheel colides with barrier or the side of the wall
        self.collide =  False

        #flag to indicate if the robot reach finsih line
        self.reach_finish = False

        #flag to indicate if the robot already reach maximum time
        self.time_out = False

        #intialize all needed value
        self.calc_robot_pos()
        self.calc_wheel_pos()
        self.calc_rays_pos()
        
        #for debuggin purpose
        self.proximity_penalty = 0
        self.moving_penalty = 0
        self.position_reward = 0



    def trans_robot2Normal(self,x0,y0,theta,xR,yR):
        """
        Transformation matrix from robot frame coordinate to world coordinate.

        Parameters:
            x0 (float): x coordinate of the robot origin (middle of the body) relative to the world origin 
            y0 (float): y coordinate of the robot origin (middle of the body) relative to the world origin 
            theta (float): rotation of the robot body around z axis of the world coordinate (rad)
            xR (float): x coordinate relative to the robot origin 
            yR (float): y coordinate relative to the robot origin 

        Returns:
            np.ndarray: Numpy array containing:
                - x (float): x coordinate of the point relative to the world origin 
                - y (float): y coordinate of the point relative to the world origin 

        """

        x = x0 + xR*np.cos(theta) - yR*np.sin(theta)
        y = y0 + xR*np.sin(theta) + yR*np.cos(theta)

        return [x,y]


    def calc_robot_pos(self):
        """
        Function to calculatethe position of each edge of the robot
        """
        #calculate the position of each corner rlative to the middle of the robot's body 
        xR_left = -self.robot_width/2
        xR_right = self.robot_width/2
        yR_bottom = -self.robot_length/2
        yR_top = self.robot_length/2

        #calculate coordinate pair of each corner relative to the world coordinate
        bottom_left = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xR_left,yR_bottom)
        top_left = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xR_left,yR_top)
        top_right = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xR_right,yR_top)
        bottom_right = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xR_right,yR_bottom)

        self.robot_pos_buffer.append([bottom_left,top_left,top_right,bottom_right])


    def calc_wheel_pos(self):
        """
        Function to calculate each edge of the left and right wheel
        """
        #calculate the position of each corner of the wheels relative to the robot coordinate
        xRL_left = -self.robot_width/2-self.wheel_width #x left of left wheel
        xRL_right = -self.robot_width/2 #x right of left wheel
        yRL_bottom = -self.wheel_length/2 #y bottom of left wheel
        yRL_top = self.wheel_length/2 #y top of left wheel

        xRR_left = self.robot_width/2 #x left of right wheel
        xRR_right = self.robot_width/2+self.wheel_width #x right of right wheel
        yRR_bottom = -self.wheel_length/2 #y bottom of right wheel
        yRR_top = self.wheel_length/2 #y top of right wheel

        #calculate coordinate pair of each wheel's corner relative to the world coordinate
        bottom_leftL = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xRL_left,yRL_bottom)
        top_leftL = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xRL_left,yRL_top)
        top_rightL = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xRL_right,yRL_top)
        bottom_rightL = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xRL_right,yRL_bottom)

        bottom_leftR = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xRR_left,yRR_bottom)
        top_leftR = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xRR_left,yRR_top)
        top_rightR = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xRR_right,yRR_top)
        bottom_rightR = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,xRR_right,yRR_bottom)

        self.wheelL_pos_buffer.append([bottom_leftL,top_leftL,top_rightL,bottom_rightL])
        self.wheelR_pos_buffer.append([bottom_leftR,top_leftR,top_rightR,bottom_rightR])

    
    def calc_rays_pos(self):
        """
        Function to calculate position of each LIDAR rays
        """
        theta_list = []
        lidar_pos_pair = []
        dTheta_lidar =  360/self.num_lidarRays
        i = 0

        while i < self.num_lidarRays:
            ray_pos = np.deg2rad(0.0 + i*dTheta_lidar)
            theta_list.append(ray_pos) 
            i=i+1
        
        for theta in theta_list:
            x1 = 0.0
            y1 = 0.0

            x2 = x1 + self.lidar_range*np.sin(theta)
            y2 = y1 + self.lidar_range*np.cos(theta)

            p1 = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,x1,y1)
            p2 = self.trans_robot2Normal(self.x_real,self.y_real,self.theta_read,x2,y2) 

            coor_pair = [p1,p2]
            lidar_pos_pair.append(coor_pair)
        
        self.lidar_pos_pair_buffer.append(lidar_pos_pair)

    
    def calc_lidar_distance(self):
        """
        get the distance data from lidar in virtual and in real environment
        """
        self.lidar_dis_read = []

        for ray_start,ray_end in self.lidar_pos_pair_buffer[-1]:
            distance = self.lidar_range
            
            #check collision with barrier
            for barrier, radius in zip(self.barrier_pos, self.barrier_radius):
                new_distance = self.ray_intersects_circle(ray_start,ray_end,barrier,radius)
                if new_distance < distance:
                    distance = new_distance
            
            #check left border
            new_distance = self.ray_intersects_line(ray_start,ray_end,[0.0,0.0],[0.0,self.arena_length],self.lidar_range)
            if new_distance < distance:
                distance = new_distance

            #check right border
            new_distance = self.ray_intersects_line(ray_start,ray_end,[self.arena_width,self.arena_length],[self.arena_width,0.0],self.lidar_range)
            if new_distance < distance:
                distance = new_distance
            
            #check bottom border
            new_distance = self.ray_intersects_line(ray_start,ray_end,[self.arena_width,0.0],[0.0,0.0],self.lidar_range)
            if new_distance < distance:
                distance = new_distance

            #check top border
            new_distance = self.ray_intersects_line(ray_start,ray_end,[0.0,self.arena_length],[self.arena_width,self.arena_length],self.lidar_range)
            if new_distance < distance:
                distance = new_distance
            
            
            self.lidar_dis_read.append(distance)

        self.lidar_dis_buffer.append(self.lidar_dis_read)


    def ray_intersects_line(self, ray_start, ray_end, barrier_start, barrier_end,ray_max):
        """ 
        Check if a ray intersects a line segment

        Parameters:
            - ray_start (list of float): list of x,y coordinates of the starting point of the ray
            - ray_end (list of float): list of x,y coordinates of the end poit of the ray
            - barrier_start (list of float): list of x,y coordinates of the starting point of the barrier
            - barrier_end (list of float): list of x,y coordinates of the end point of the barrier
        Returns:
            - distance (float): distance of the reading of the LIDAR. Will return the maximum range value if there is no object detected
        """
       
        x1, y1 = ray_start
        x2, y2 = ray_end
        x3, y3 = barrier_start
        x4, y4 = barrier_end

        # Compute intersection using determinant method
        denom = (x2 - x1) * (y3 - y4) - (y2 - y1) * (x3 - x4)
        if denom == 0:
            return ray_max  # Parallel lines

        # Compute intersection point
        t = ((x3 - x1) * (y3 - y4) - (y3 - y1) * (x3 - x4)) / denom
        u = ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / denom

        # Check if intersection is within bounds
        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            
            distance = np.sqrt((ix-x1)**2 + (iy-y1)**2)
            return distance

        return ray_max



    def ray_intersects_circle(self, ray_start, ray_end, circle_center, circle_radius):
        """ 
        Check if a ray intersects a circle

        Parameters:
            - ray_start (list of float): list of x,y coordinates of the starting point of the ray
            - ray_end (list of float): list of x,y coordinates of the end poit of the ray
            - circle_center (list of float): list of x,y coordinates of the center of the circle
            - circle_radius (float): radius of the circle
        Returns:
            - distance (float): distance of the reading of the LIDAR. Will return the maximum range value if there is no object detected
        """
        x1, y1 = ray_start
        dx = ray_end[0] - ray_start[0]
        dy = ray_end[1] - ray_start[1]

        a = dx**2 + dy**2
        b = 2 * dx * (x1 - circle_center[0]) + 2 * dy * (y1 - circle_center[1])
        c = (x1 - circle_center[0])**2 + (y1 - circle_center[1])**2 - circle_radius**2

        discriminant = b**2 - 4 * a * c
        
        if discriminant < 0:
            return self.lidar_range  # No intersection

        t1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        t2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        if 0<=t1<=1:
            x = x1 + dx*t1
            y = y1 + dy*t1
            distance = np.sqrt((x-x1)**2 + (y-y1)**2)
        
        elif 0<=t2<=1:
            x = x1 + dx*t2
            y = y1 + dy*t2
            distance = np.sqrt((x-x1)**2 + (y-y1)**2)

        else:
            distance = self.lidar_range

        return distance


    
    def check_line_collision(self, robot_start, robot_end, boundary_start, boundary_end):
        """ 
        Check if a line in robot intersects other line segment in arena

        Parameters:
            - robot_start (list of float): list of x,y coordinates of the starting point of the robot/wheel side
            - robot_end (list of float): list of x,y coordinates of the end poit of the robot/wheel side
            - boundary_start (list of float): list of x,y coordinates of the starting point of the boundary
            - boundary_end (list of float): list of x,y coordinates of the end point of the boundary

        Returns:
            - bool: True if there is an intersection, False otherwise
        """
       
        x1, y1 = robot_start
        x2, y2 = robot_end
        x3, y3 = boundary_start
        x4, y4 = boundary_end

        # Compute intersection using determinant method
        denom = (x2 - x1) * (y3 - y4) - (y2 - y1) * (x3 - x4)
        if denom == 0:
            return False

        # Compute intersection point
        t = ((x3 - x1) * (y3 - y4) - (y3 - y1) * (x3 - x4)) / denom
        u = ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / denom

        # Check if intersection is within bounds
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True

        return False
 
    
    def check_circle_collision(self, robot_start, robot_end, circle_center, circle_radius):
        """
        Check if a line segment intersects a circle

        Parameters:
            - robot_start (list of float): list of x,y coordinates of the starting point of the robot/wheel side
            - robot_end (list of float): list of x,y coordinates of the end poit of the robot/wheel side
            - circle_center (list of float): list of x,y coordinates of the center of the circle
            - circle_radius (float): radius of the circle
        
        Returns:
            - bool: True if there is an intersection, False otherwise
        """
        x1, y1 = robot_start
        dx = robot_start[0] - robot_end[0]
        dy = robot_end[1] - robot_start[1]

        a = dx**2 + dy**2
        b = 2 * dx * (x1 - circle_center[0]) + 2 * dy * (y1 - circle_center[1])
        c = (x1 - circle_center[0])**2 + (y1 - circle_center[1])**2 - circle_radius**2

        discriminant = b**2 - 4 * a * c

        # Check if the discriminant is negative
        if discriminant < 0:
            return False

        t1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        t2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        if 0<=t1<=1 or 0<=t2<=1:
            return True
        
        else:  
            return False


    
    def robot_collision_check(self):
        """
        Check if the robot collide with the side of the map or with the barrier
        """

        bottom_left,top_left,top_right,bottom_right = self.robot_pos_buffer[-1]

        border_left = [[0.0,0.0],[0.0,self.arena_length]]
        border_right = [[self.arena_width,self.arena_length],[self.arena_width,0.0]]
        border_bottom = [[self.arena_width,0.0],[0.0,0.0]]
        border_top = [[0.0,self.arena_length],[self.arena_width,self.arena_length]]

        border_sides = [border_left,border_right,border_bottom, border_top]

        robot_left = [bottom_left,top_left]
        robot_top = [top_left,top_right]
        robot_right = [top_right,bottom_right]
        robot_bottom = [bottom_right,bottom_left]

        robot_sides =[robot_left,robot_top,robot_right,robot_bottom]

        for sideR in robot_sides:
            if self.collide:
                break
            #check collison with border
            for sideB in border_sides:
                if self.collide:
                    break
                
                self.collide = self.check_line_collision(sideR[0],sideR[1],sideB[0],sideB[1])

            #check collision with barrier
            for barrier, radius in zip(self.barrier_pos,self.barrier_radius):
                if self.collide:
                    break
                self.collide = self.check_circle_collision(sideR[0],sideR[1],barrier,radius)



    def wheel_collision_check(self):
        """
        Check if the robot collide with the side of the map or with the barrier
        """

        border_left = [[0.0,0.0],[0.0,self.arena_length]]
        border_right = [[self.arena_width,self.arena_length],[self.arena_width,0.0]]
        border_bottom = [[self.arena_width,0.0],[0.0,0.0]]
        border_top = [[0.0,self.arena_length],[self.arena_width,self.arena_length]]

        border_sides = [border_left,border_right,border_bottom, border_top]

        bottom_leftL,top_leftL,top_rightL,bottom_rightL = self.wheelL_pos_buffer[-1]

        wheelL_left = [bottom_leftL,top_leftL]
        wheelL_top = [top_leftL,top_rightL]
        wheelL_right = [top_rightL,bottom_rightL]
        wheelL_bottom = [bottom_rightL,bottom_leftL]

        wheelL_sides =[wheelL_left,wheelL_top,wheelL_right,wheelL_bottom]

        bottom_leftR,top_leftR,top_rightR,bottom_rightR = self.wheelR_pos_buffer[-1]
        
        wheelR_left = [bottom_leftR,top_leftR]
        wheelR_top = [top_leftR,top_rightR]
        wheelR_right = [top_rightR,bottom_rightR]
        wheelR_bottom = [bottom_rightR,bottom_leftR]

        wheelR_sides =[wheelR_left,wheelR_top,wheelR_right,wheelR_bottom]

        #check copllision for left wheel
        for sideWL in wheelL_sides:
            if self.collide:
                break
            #check collison with border
            for sideB in border_sides:
                if self.collide:
                    break
                self.collide = self.check_line_collision(sideWL[0],sideWL[1],sideB[0],sideB[1])

            #check collision with barrier
            for barrier, radius in zip(self.barrier_pos,self.barrier_radius):
                if self.collide:
                    break
                self.collide = self.check_circle_collision(sideWL[0],sideWL[1],barrier,radius)


        #check copllision for right wheel
        for sideWR in wheelR_sides:
            if self.collide:
                break
            #check collison with border
            for sideB in border_sides:
                if self.collide:
                    break
                self.collide = self.check_line_collision(sideWR[0],sideWR[1],sideB[0],sideB[1])
            
            #check collision with barrier
            for barrier, radius in zip(self.barrier_pos, self.barrier_radius):
                if self.collide:
                    break
                self.collide = self.check_circle_collision(sideWR[0],sideWR[1],barrier,radius)




    def check_finsih(self):
        """
        Check whether part of the robot already cross finsih line
        """
        dis_to_finish = np.sqrt((self.x_real - self.target_pos[0])**2 + (self.y_real - self.target_pos[1])**2)
        if dis_to_finish < 0.05:
            self.reach_finish = True


    def check_timeout(self):
        """
        Function to check whether the robot already reach the maximum time
        """

        if self.time_passed > self.max_time:
            self.time_out = True


    def divided_lidar_batch(self):
        """
        Function to divide LIDAR into several batch.
        """
        
        self.lidar_dis_min = []
        batch_data = []
        i = 0
        data_length = int(self.num_lidarRays/self.lidar_batch_num)

        while i < self.lidar_batch_num:
            
            #if it is the last batch, take all the rest of ray into the same batch
            if i+1 == self.lidar_batch_num:
                batch_data.append(self.lidar_dis_read[i*data_length:])

            else:
                batch_data.append(self.lidar_dis_read[ i*data_length : (i+1)*data_length ])
            
            i=i+1
        
        for batch in batch_data:
            self.lidar_dis_min.append(min(batch))



    def calculate_reward(self, prev_distance):
        """
        Function to calculate reward for each step
        
        Returns:
            - rewards (float): reward of the step
        """
        
        proximity_penalty = 0
        ray_length = self.lidar_range
        
        for read in self.lidar_dis_read:
            proximity_penalty -= ((ray_length - read)/ray_length)*(1/self.lidar_batch_num)
            

        max_linear_velocity = self.max_speed*self.wheel_length/2
        current_distance = np.sqrt((self.x_real - self.target_pos[0])**2 + (self.y_real - self.target_pos[1])**2)
        dDistance = prev_distance - current_distance
        running_penalty = dDistance/(max_linear_velocity*self.dt*self.step_num)

        
        self.moving_penalty = running_penalty
        self.proximity_penalty = proximity_penalty

        if self.collide:
            reward =  -5
            self.position_reward = reward

        else:
            reward = 3*proximity_penalty + 7*running_penalty
            self.position_reward = reward
        
        return  reward



    def reset_env(self):
        """
        function to reset environment
        
        Parameters:
            - random_barrier (bool): To choose to randomize the barrier or not (default value = False).
        Returns:
            - returned_obs (list of float): current observation of the robot
        """
        
        self.my_robot.reset()
        #positon and state of robot based on the reading from the robot model
        self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read = self.my_robot.get_state()

        #position of the robot in real and virtual based on our coordinate taken from the middle of the robot
        self.x_real = self.start_x
        self.y_real = self.start_y

        self.time_passed = 0

        #number of lidar rays place to save lidar reading
        self.lidar_dis_read = [] #real world reading
        self.lidar_dis_min = []

        #buffer for replay in pygame
        self.lidar_pos_pair_buffer = []
        self.robot_pos_buffer = []
        self.wheelL_pos_buffer = []
        self.wheelR_pos_buffer = []
        self.lidar_dis_buffer = []


        #reset end flag
        self.episode_end = False

        #flag to indicate whether the robot or wheel colides with barrier or the side of the wall
        self.collide =  False

        #flag to indicate if the robot reach finsih line
        self.reach_finish = False

        #flag to indicate if the robot already reach maximum time
        self.time_out = False

        #intialize all needed value
        self.calc_robot_pos()
        self.calc_wheel_pos()
        self.calc_rays_pos()
        self.calc_lidar_distance()
        self.divided_lidar_batch()

        returned_obs = [self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read,self.time_passed] + self.lidar_dis_min
        
        return returned_obs



    def step(self,wL, wR, user_test=False):
        """
        Function compute the next state of the robot
        
        Parameters:
            - wL (float): left wheel rotation speed
            - wR (float): right wheel rotation speeds
        
        Returns:
            - returned_values (list of float): list of important observation,reward and end episode flag
        """

        if self.episode_end == False:
            reward = 0
            
            i = 0
            prev_distance = np.sqrt((self.x_real - self.target_pos[0])**2 + (self.y_real - self.target_pos[1])**2)
            while i<self.step_num:
                self.my_robot.computeKinematic(wR,wL)
                self.time_passed = self.time_passed + self.dt
                #positon and state of robot based on the reading from the robot model
                self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read = self.my_robot.get_state()

                #position of the robot in real and virtual based on our coordinate taken from the middle of the robot
                self.x_real = self.x_real + self.dx_read*self.dt
                self.y_real = self.y_real + self.dy_read*self.dt
                
                #calculate new pos
                self.calc_robot_pos()
                self.calc_wheel_pos()
                self.calc_rays_pos()
                self.calc_lidar_distance()
                self.divided_lidar_batch()

                if not user_test:
                    #check distance of LIDAR,collision, and whether the robot already touch finsih line
                    self.robot_collision_check()
                    self.wheel_collision_check()
                    self.check_timeout()
                    self.check_finsih()

                if self.collide or self.reach_finish or self.time_out:
                    break
                
                i += 1

            reward = self.calculate_reward(prev_distance=prev_distance)
            
            if self.collide or self.reach_finish or self.time_out:
                self.episode_end = True

            returned_obs = [self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read,self.time_passed] + self.lidar_dis_min
            
            returned_values = [returned_obs,reward,self.episode_end]

            return returned_values
        
        else:
            reward = 0.0
            returned_obs = [self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read,self.time_passed] + self.lidar_dis_min
            
            
            returned_values = [returned_obs,reward,self.episode_end]

            return returned_values
        
    def test_arena(self):
        """
        Function to visualize the arena
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0,self.arena_width) 
        ax.set_ylim(0,self.arena_length)
        ax.set_aspect('equal')
        
        #draw barrier
        for barrier, radius in zip(self.barrier_pos,self.barrier_radius):
            circle = Circle((barrier[0], barrier[1]), radius, color='r', fill=False)
            ax.add_artist(circle)

        #draw target
        target_x, target_y = self.target_pos
        ax.plot(target_x, target_y, marker='x', color='g', markersize=8, label="Target")  # Green point for the target

        #draw lidar
        lidar_lines = []
        for (ray_start,ray_end), distance in zip(self.lidar_pos_pair_buffer[-1],self.lidar_dis_read):
            dx = ray_end[0] - ray_start[0]
            dy = ray_end[1] - ray_start[1]
            t = distance/self.lidar_range
            adjusted_end = [ray_start[0] + t*dx, ray_start[1] + t*dy]
            lidar_lines.append(ax.plot([ray_start[0],adjusted_end[0]], [ray_start[1],adjusted_end[1]],'y-',lw=0.5))

        #draw robot and wheel
        robot_patch = Polygon(self.robot_pos_buffer[-1], closed=True, fill=None, edgecolor='g', lw=2)
        ax.add_patch(robot_patch)

        wheelL_patch = Polygon(self.wheelL_pos_buffer[-1], closed=True, fill=None, edgecolor='b', lw=2)
        ax.add_patch(wheelL_patch)

        wheelR_patch = Polygon(self.wheelR_pos_buffer[-1], closed=True, fill=None, edgecolor='b', lw=2)
        ax.add_patch(wheelR_patch)

        plt.show()




    def user_control(self):
        """
        Function to visualize the arena and control the robot
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0,self.arena_width) 
        ax.set_ylim(0,self.arena_length)
        ax.set_aspect('equal')
        
        #draw barrier
        for barrier, radius in zip(self.barrier_pos,self.barrier_radius):
            circle = Circle((barrier[0], barrier[1]), radius, color='r', fill=False)
            ax.add_artist(circle)

        #draw target
        target_x, target_y = self.target_pos
        ax.plot(target_x, target_y, marker='x', color='g', markersize=8, label="Target")  # Green point for the target

        #draw lidar
        lidar_lines = []
        for (ray_start,ray_end), distance in zip(self.lidar_pos_pair_buffer[-1],self.lidar_dis_read):
            dx = ray_end[0] - ray_start[0]
            dy = ray_end[1] - ray_start[1]
            t = distance/self.lidar_range
            adjusted_end = [ray_start[0] + t*dx, ray_start[1] + t*dy]
            line, = ax.plot([ray_start[0],adjusted_end[0]], [ray_start[1],adjusted_end[1]],'y-',lw=0.5)
            lidar_lines.append(line)

        #draw robot and wheel
        robot_patch = Polygon(self.robot_pos_buffer[-1], closed=True, fill=None, edgecolor='g', lw=2)
        ax.add_patch(robot_patch)

        wheelL_patch = Polygon(self.wheelL_pos_buffer[-1], closed=True, fill=None, edgecolor='b', lw=2)
        ax.add_patch(wheelL_patch)

        wheelR_patch = Polygon(self.wheelR_pos_buffer[-1], closed=True, fill=None, edgecolor='b', lw=2)
        ax.add_patch(wheelR_patch)


         # Variables to control robot speed
        wL, wR = 0.0, 0.0  # Left and right wheel speeds
        key_pressed = False
        reward_total = 0

        # Add text annotations for robot state
        text_x_real = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='black')
        text_y_real = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10, color='black')
        text_theta_read = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=10, color='black')
        text_x_read = ax.text(0.02, 0.80, '', transform=ax.transAxes, fontsize=10, color='black')
        text_y_read = ax.text(0.02, 0.75, '', transform=ax.transAxes, fontsize=10, color='black')
        text_proximty_penalty = ax.text(0.02, 0.70, '', transform=ax.transAxes, fontsize=10, color='black')
        text_moving_penalty = ax.text(0.02, 0.65, '', transform=ax.transAxes, fontsize=10, color='black')
        text_position_reward = ax.text(0.02, 0.60, '', transform=ax.transAxes, fontsize=10, color='black')
        text_reward_total = ax.text(0.02, 0.55, '', transform=ax.transAxes, fontsize=10, color='black')


        def update_plot(frame):
            """
            Update the robot, wheel, and LIDAR positions on the plot.
            """
            # Perform a simulation step with the current wheel speeds
            nonlocal wL, wR, key_pressed, reward_total

            if not key_pressed:
                wL, wR = 0.0, 0.0

            returns = self.step(wL, wR, user_test=True)
            reward_total += returns[1]


            key_pressed = False

            # Update robot and wheel positions
            robot_patch.set_xy(self.robot_pos_buffer[-1])
            wheelL_patch.set_xy(self.wheelL_pos_buffer[-1])
            wheelR_patch.set_xy(self.wheelR_pos_buffer[-1])

            # Update LIDAR rays
            for line, (ray_start, ray_end), distance in zip(lidar_lines, self.lidar_pos_pair_buffer[-1], self.lidar_dis_read):
                dx = ray_end[0] - ray_start[0]
                dy = ray_end[1] - ray_start[1]
                t = distance / self.lidar_range
                adjusted_end = [ray_start[0] + t * dx, ray_start[1] + t * dy]
                line.set_data([ray_start[0], adjusted_end[0]], [ray_start[1], adjusted_end[1]])

            # Update text annotations
            text_x_real.set_text(f"x_real: {self.x_real:.2f}")
            text_y_real.set_text(f"y_real: {self.y_real:.2f}")
            text_theta_read.set_text(f"theta_read: {np.rad2deg(self.theta_read):.2f}°")
            text_x_read.set_text(f"x_read: {self.x_read:.2f}")
            text_y_read.set_text(f"y_read: {self.y_read:.2f}")
            text_proximty_penalty.set_text(f"Proximity Penalty: {self.proximity_penalty:.2f}")
            text_moving_penalty.set_text(f"Moving Penalty: {self.moving_penalty:.2f}")
            text_position_reward.set_text(f"Position Reward: {self.position_reward:.2f}")
            text_reward_total.set_text(f"Total Reward: {reward_total:.2f}")

            texts = [text_x_real, text_y_real, text_theta_read, text_x_read, text_y_read, text_proximty_penalty, 
                     text_moving_penalty, text_position_reward, text_reward_total]
            
            return [robot_patch, wheelL_patch, wheelR_patch] + lidar_lines + texts

        def on_key(event):
            """
            Handle keyboard input to control the robot.
            """
            nonlocal wL, wR, key_pressed

            key_pressed = True

            if event.key == 'up':  # Move forward
                wL = self.max_speed
                wR = self.max_speed
            elif event.key == 'down':  # Move backward
                wL = -self.max_speed
                wR = -self.max_speed
            elif event.key == 'left':  # Turn left
                wL = -self.max_speed
                wR = self.max_speed
            elif event.key == 'right':  # Turn right
                wL = self.max_speed
                wR = -self.max_speed
    
        # Connect the key press event to the handler
        fig.canvas.mpl_connect('key_press_event', on_key)

        # Create the animation
        ani = animation.FuncAnimation(fig, update_plot, frames=None, interval=10, blit=True)

        plt.show()



    
    def run_replay(self):
        """
        Function to replay the movement of the robot
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0,self.arena_width) 
        ax.set_ylim(0,self.arena_length)
        ax.set_aspect('equal')
        
        #draw barrier
        for barrier, radius in zip(self.barrier_pos,self.barrier_radius):
            circle = Circle((barrier[0], barrier[1]), radius, color='r', fill=False)
            ax.add_artist(circle)

        #draw target
        target_x, target_y = self.target_pos
        ax.plot(target_x, target_y, marker='x', color='g', markersize=8, label="Target")  # Green point for the target

        #draw lidar
        lidar_lines = []
        for (ray_start,ray_end), distance in zip(self.lidar_pos_pair_buffer[0],self.lidar_dis_buffer[0]):
            dx = ray_end[0] - ray_start[0]
            dy = ray_end[1] - ray_start[1]
            t = distance/self.lidar_range
            adjusted_end = [ray_start[0] + t*dx, ray_start[1] + t*dy]
            line, = ax.plot([ray_start[0],adjusted_end[0]], [ray_start[1],adjusted_end[1]],'y-',lw=0.5)
            lidar_lines.append(line)

        #draw robot and wheel
        robot_patch = Polygon(self.robot_pos_buffer[0], closed=True, fill=None, edgecolor='g', lw=2)
        ax.add_patch(robot_patch)

        wheelL_patch = Polygon(self.wheelL_pos_buffer[0], closed=True, fill=None, edgecolor='b', lw=2)
        ax.add_patch(wheelL_patch)

        wheelR_patch = Polygon(self.wheelR_pos_buffer[0], closed=True, fill=None, edgecolor='b', lw=2)
        ax.add_patch(wheelR_patch)



        def update_plot(frame):
            """
            Update the robot, wheel, and LIDAR positions on the plot.
            """    

            # Update robot and wheel positions
            robot_patch.set_xy(self.robot_pos_buffer[frame])
            wheelL_patch.set_xy(self.wheelL_pos_buffer[frame])
            wheelR_patch.set_xy(self.wheelR_pos_buffer[frame])

            # Update LIDAR rays
            for line, (ray_start, ray_end), distance in zip(lidar_lines, self.lidar_pos_pair_buffer[frame], self.lidar_dis_buffer[frame]):
                dx = ray_end[0] - ray_start[0]
                dy = ray_end[1] - ray_start[1]
                t = distance / self.lidar_range
                adjusted_end = [ray_start[0] + t * dx, ray_start[1] + t * dy]
                line.set_data([ray_start[0], adjusted_end[0]], [ray_start[1], adjusted_end[1]])


            return [robot_patch, wheelL_patch, wheelR_patch] + lidar_lines


        # Create the animation
        ani = animation.FuncAnimation(fig, update_plot, frames=len(self.robot_pos_buffer), interval=10, blit=True)

        plt.show()