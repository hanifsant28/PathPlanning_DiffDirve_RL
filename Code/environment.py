import numpy as np
import pygame 
import random
from kinematic import DifDriveKinematic


class environment:
    def __init__(self,
                 lane_num,
                 screen_width,
                 screen_height,
                 arena_width,
                 robot_width,
                 robot_length,
                 num_lidarRay,
                 num_lidarBatch,
                 min_dBarrier,
                 max_dBarrier,
                 max_speed,
                 max_time,
                 dt,
                 step_num):
        
        """
        Initialize the environment for the robot.
        
        Parameters:
            lane_num (int): Number of desired lane.
            screen_width (int): width of the animation screen.
            screen_height (int): height of the animation screen.
            arena_width (float): actual width of the arena.
            robot_width (float): actual robot width.
            robot_length (float): actual robot length.
            num_lidarRay (int): number of desired array.
            num_lidarBatch (int): number of batch for lidar reading.
            min_dBarrier (float): minimum distance between barrier.
            max_dBarrier (float): maximum distance between barrier.
            max_speed (float): maximum speed of the robot in (m/s).
            dt (float): time increment for simulation.
            step_num (int): number of step every time action taken.
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


        
        self.lane_num = lane_num
        self.barrier_pos_pair = [] #list of pair of coordinates of the left side of the barrier
        self.screen_width = screen_width
        self.screen_height =  screen_height
        self.arena_width = arena_width #arena widht in real world
        self.alpha = screen_width/arena_width #scale from real to virtual width

        self.lane_width = screen_width/self.lane_num #width in pixel
        self.min_dBarrier = min_dBarrier*self.alpha
        self.max_dBarrier = max_dBarrier*self.alpha
        
        self.step_num = step_num

        self.max_speed = max_speed
        self.dt = dt
        self.max_time = max_time
        self.time_passed = 0

        #size calculation for robot
        self.robot_width_real = robot_width
        self.robot_length_real = robot_length

        self.robot_width_vir = self.robot_width_real * self.alpha #robot width virtual
        self.robot_length_vir = self.robot_length_real *self.alpha #robot height virtual

        #calculation of wheel size
        wheel_width_real = self.robot_width_real * 0.2
        wheel_length_real = self.robot_length_real * 0.333334

        self.wheel_width_vir = wheel_width_real * self.alpha #wheel width virtual
        self.wheel_length_vir = wheel_length_real * self.alpha #wheel height virtual

        #intialize robot object
        wheel_radius = self.wheel_length_vir/2
        self.my_robot = DifDriveKinematic(wheel_radius,self.robot_width_real,self.dt)

        #positon and state of robot based on the reading from the robot model
        self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read = self.my_robot.get_state()

        #position of the robot in real and virtual based on our coordinate taken from the middle of the robot
        self.x_real = arena_width/2
        self.y_real = self.robot_length_real/2 + 0.05

        self.x_vir = self.x_real*self.alpha
        self.y_vir = self.y_real*self.alpha

        #number of lidar rays place to save lidar reading
        self.lidar_batch_num = num_lidarBatch
        self.num_lidarRays = num_lidarRay
        self.lidar_dis_read = [] #real world reading
        self.lidar_dis_vir = [] #vir world reading

        #buffer for replay in pygame
        self.robot_pos_buffer = []
        self.wheelL_pos_buffer = []
        self.wheelR_pos_buffer = []
        self.lidar_pos_pair_buffer = []

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
        self.randomize_barrier_pos()
        self.calc_rays_pos()


    def trans_World2pygame(self,x,y):
        """
        Translate coordinate from world coordinate ((0,0) at the bottom left corner) 
        to pygame coordinate ((0,0) at the bottom right corner).
        
        Parameters:
            x (float): x coordinate of world coordinate
            y (float): y coordinate of world coordinate
        
        Returns:
            np.ndarray: Numpy array containing:
                - x (float): x coordinate of pygame system
                - y (float): y coordinate of pygame system
        """

        return [x,(self.screen_height-y)]


    def trans_robot2Normal(self,x0,y0,theta,xR,yR):
        """
        Transformation matrix from robot frame coordinate to world coordinate 
        in normal unit.

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
        xR_left = -self.robot_width_vir/2
        xR_right = self.robot_width_vir/2
        yR_bottom = -self.robot_length_vir/2
        yR_top = self.robot_length_vir/2

        #calculate coordinate pair of each corner relative to the world coordinate
        bottom_left = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xR_left,yR_bottom)
        top_left = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xR_left,yR_top)
        top_right = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xR_right,yR_top)
        bottom_right = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xR_right,yR_bottom)

        self.robot_pos_buffer.append([bottom_left,top_left,top_right,bottom_right])


    def calc_wheel_pos(self):
        """
        Function to calculate each edge of the left and right wheel
        """
        #calculate the position of each corner of the wheels relative to the robot coordinate
        xRL_left = -self.robot_width_vir/2-self.wheel_width_vir #x left of left wheel
        xRL_right = -self.robot_width_vir/2 #x right of left wheel
        yRL_bottom = -self.wheel_length_vir/2 #y bottom of left wheel
        yRL_top = self.wheel_length_vir/2 #y top of left wheel

        xRR_left = self.robot_width_vir/2 #x left of right wheel
        xRR_right = self.robot_width_vir/2+self.wheel_width_vir #x right of right wheel
        yRR_bottom = -self.wheel_length_vir/2 #y bottom of right wheel
        yRR_top = self.wheel_length_vir/2 #y top of right wheel

        #calculate coordinate pair of each wheel's corner relative to the world coordinate
        bottom_leftL = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xRL_left,yRL_bottom)
        top_leftL = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xRL_left,yRL_top)
        top_rightL = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xRL_right,yRL_top)
        bottom_rightL = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xRL_right,yRL_bottom)

        bottom_leftR = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xRR_left,yRR_bottom)
        top_leftR = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xRR_left,yRR_top)
        top_rightR = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xRR_right,yRR_top)
        bottom_rightR = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,xRR_right,yRR_bottom)

        self.wheelL_pos_buffer.append([bottom_leftL,top_leftL,top_rightL,bottom_rightL])
        self.wheelR_pos_buffer.append([bottom_leftR,top_leftR,top_rightR,bottom_rightR])


    def randomize_barrier_pos(self):
        """
        Function to randomize the position of the barrier
        """

        barrier_num = int(self.screen_height/(self.min_dBarrier)) #max number of barrier
        n = 1
        x1= 0.0
        y1 = 0.0
        y2 = 0.0
        x2 = 0.0
        while n <= barrier_num:
            
            lane = random.randint(0,self.lane_num) #lane where the barrier is located

            x1 = 0.0 + lane*self.lane_width
            y1 = y2 + random.uniform(self.min_dBarrier,self.max_dBarrier)

            min_width = self.lane_width*(0-lane)+(self.wheel_width_vir*2+self.robot_width_vir)*2
            max_width = self.lane_width*(self.lane_num-lane)-(self.wheel_width_vir*2+self.robot_width_vir)*2
            x2 = x1+random.uniform(min_width,max_width)
            y2 = y1

            coor_pair = [[x1,y1],[x2,y2]] #coordinate pair of barrier from start to end

            if(y1 < self.screen_height):
                self.barrier_pos_pair.append(coor_pair)
            
            n=n+1

    
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
        
        ray_length = 1.5 * self.robot_length_vir

        for theta in theta_list:
            x1 = 0.0
            y1 = 0.0

            x2 = x1 + ray_length*np.sin(theta)
            y2 = y1 + ray_length*np.cos(theta)

            p1 = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,x1,y1)
            p2 = self.trans_robot2Normal(self.x_vir,self.y_vir,self.theta_read,x2,y2) 

            coor_pair = [p1,p2]
            lidar_pos_pair.append(coor_pair)
        
        self.lidar_pos_pair_buffer.append(lidar_pos_pair)

    
    def calc_lidar_distance(self):
        """
        get the distance data from lidar in virtual and in real environment
        """
        self.lidar_dis_read = []
        ray_length = 1.5 * self.robot_length_vir

        for ray_start,ray_end in self.lidar_pos_pair_buffer[-1]:
            distance = ray_length
            
            #check barrier
            for barrier_start,barrier_end in self.barrier_pos_pair:
                new_distance = self.ray_intersects_line(ray_start,ray_end,barrier_start,barrier_end,ray_length)
                if new_distance < distance:
                    distance = new_distance
            
            #check left border
            new_distance = self.ray_intersects_line(ray_start,ray_end,[0.0,0.0],[0.0,self.screen_height],ray_length)
            if new_distance < distance:
                distance = new_distance

            #check right border
            new_distance = self.ray_intersects_line(ray_start,ray_end,[self.screen_width,self.screen_height],[self.screen_width,0.0],ray_length)
            if new_distance < distance:
                distance = new_distance
            
            #check bottom border
            new_distance = self.ray_intersects_line(ray_start,ray_end,[self.screen_width,0.0],[0.0,0.0],ray_length)
            if new_distance < distance:
                distance = new_distance
            
            self.lidar_dis_vir.append(distance)

            real_distance = distance*(1/self.alpha)
            self.lidar_dis_read.append(real_distance)


    def ray_intersects_line(self, ray_start, ray_end, barrier_start, barrier_end,ray_max):
        """ 
        Check if a ray intersects a line segment

        Parameters:
            - ray_start (list of float): list of x,y coordinates of the starting point of the ray
            - ray_end (list of float): list of x,y coordinates of the end poit of the ray
            - barrier_start (list of float): list of x,y coordinates of the starting point of the barrier
            - barrier_end (list of float): list of x,y coordinates of the end point of the barrier
        Returns:
            - distance (float):distance of the reading of the LIDAR. Will return the maximum range value if there is no object detected
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

    
    def check_collision(self, robot_start, robot_end, boundary_start, boundary_end):
        """ 
        Check if a line in robot intersects other line segment in arena

        Parameters:
            - robot_start (list of float): list of x,y coordinates of the starting point of the robot/wheel side
            - robot_end (list of float): list of x,y coordinates of the end poit of the robot/wheel side
            - boundary_start (list of float): list of x,y coordinates of the starting point of the boundary
            - boundary_end (list of float): list of x,y coordinates of the end point of the boundary
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
 
    
    def robot_collision_check(self):
        """
        Check if the robot collide with the side of the map or with the barrier
        """

        bottom_left,top_left,top_right,bottom_right = self.robot_pos_buffer[-1]

        border_left = [[0.0,0.0],[0.0,self.screen_height]]
        border_right = [[self.screen_width,self.screen_height],[self.screen_width,0.0]]
        border_bottom = [[self.screen_width,0.0],[0.0,0.0]]

        border_sides = [border_left,border_right,border_bottom]

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
                
                self.collide = self.check_collision(sideR[0],sideR[1],sideB[0],sideB[1])

            #check collision with barrier        
            for barrier in self.barrier_pos_pair:
                if self.collide:
                    break
                
                self.collide = self.check_collision(sideR[0],sideR[1],barrier[0],barrier[1])


    def wheel_collision_check(self):
        """
        Check if the robot collide with the side of the map or with the barrier
        """

        border_left = [[0.0,0.0],[0.0,self.screen_height]]
        border_right = [[self.screen_width,self.screen_height],[self.screen_width,0.0]]
        border_bottom = [[self.screen_width,0.0],[0.0,0.0]]

        border_sides = [border_left,border_right,border_bottom]

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
                
                self.collide = self.check_collision(sideWL[0],sideWL[1],sideB[0],sideB[1])

            #check collision with barrier        
            for barrier in self.barrier_pos_pair:
                if self.collide:
                    break
                
                self.collide = self.check_collision(sideWL[0],sideWL[1],barrier[0],barrier[1])

        #check copllision for right wheel
        for sideWR in wheelR_sides:
            if self.collide:
                break
            #check collison with border
            for sideB in border_sides:
                if self.collide:
                    break
                
                self.collide = self.check_collision(sideWR[0],sideWR[1],sideB[0],sideB[1])

            #check collision with barrier        
            for barrier in self.barrier_pos_pair:
                if self.collide:
                    break
                
                self.collide = self.check_collision(sideWR[0],sideWR[1],barrier[0],barrier[1])


    def check_finsih(self):
        """
        Check whether part of the robot already cross finsih line
        """
        for edge in self.robot_pos_buffer[-1]:
            if self.reach_finish:
                break
            elif edge[1] > self.screen_height:
                self.reach_finish = True
        
        for edge in self.wheelL_pos_buffer[-1]:
            if self.reach_finish:
                break
            elif edge[1] > self.screen_height:
                self.reach_finish = True

        for edge in self.wheelR_pos_buffer[-1]:
            if self.reach_finish:
                break
            elif edge[1] > self.screen_height:
                self.reach_finish = True


    def check_timeout(self):
        """
        Function to check whether the robot already reach the maximum time
        """

        if self.time_passed > self.max_time:
            self.time_out = True


    def draw_robot(self,idx,screen):
        """
        draw robot on pygame

        Parameters:
            - idx (int): index to an array of buffer
        """
        bottom_left,top_left,top_right,bottom_right = self.robot_pos_buffer[idx]
            
        #transform each coordinate pair to the pygame coordinate system
        bottom_left = self.trans_World2pygame(bottom_left[0],bottom_left[1])
        top_left = self.trans_World2pygame(top_left[0],top_left[1])
        top_right = self.trans_World2pygame(top_right[0],top_right[1])
        bottom_right = self.trans_World2pygame(bottom_right[0],bottom_right[1])

        #draw the square using four line by first converting the value to int 
        pos1 = [int(x) for x in bottom_left]
        pos2 = [int(x) for x in top_left]
        pos3 = [int(x) for x in top_right]
        pos4 = [int(x) for x in bottom_right]

        pygame.draw.line(screen,self.GREEN,pos1,pos2,2) #left side
        pygame.draw.line(screen,self.GREEN,pos2,pos3,2) #top side
        pygame.draw.line(screen,self.GREEN,pos3,pos4,2) #right side
        pygame.draw.line(screen,self.GREEN,pos4,pos1,2) #bottom side

    
    def draw_wheel(self,idx,screen):
        """
        draw the left and right wheel
         
        Parameters:
            - idx (int): index to an array of buffer
        """
        
        #drawing the left wheel
        bottom_left,top_left,top_right,bottom_right = self.wheelL_pos_buffer[idx]
            
        #transform each coordinate pair to the pygame coordinate system
        bottom_left = self.trans_World2pygame(bottom_left[0],bottom_left[1])
        top_left = self.trans_World2pygame(top_left[0],top_left[1])
        top_right = self.trans_World2pygame(top_right[0],top_right[1])
        bottom_right = self.trans_World2pygame(bottom_right[0],bottom_right[1])

        #draw the square using four line by first converting the value to int 
        pos1 = [int(x) for x in bottom_left]
        pos2 = [int(x) for x in top_left]
        pos3 = [int(x) for x in top_right]
        pos4 = [int(x) for x in bottom_right]

        pygame.draw.line(screen,self.BLACK,pos1,pos2,2) #left side
        pygame.draw.line(screen,self.BLACK,pos2,pos3,2) #top side
        pygame.draw.line(screen,self.BLACK,pos3,pos4,2) #right side
        pygame.draw.line(screen,self.BLACK,pos4,pos1,2) #bottom side
        
        #drawing the right wheel
        bottom_left,top_left,top_right,bottom_right = self.wheelR_pos_buffer[idx]
            
        #transform each coordinate pair to the pygame coordinate system
        bottom_left = self.trans_World2pygame(bottom_left[0],bottom_left[1])
        top_left = self.trans_World2pygame(top_left[0],top_left[1])
        top_right = self.trans_World2pygame(top_right[0],top_right[1])
        bottom_right = self.trans_World2pygame(bottom_right[0],bottom_right[1])

        #draw the square using four line by first converting the value to int 
        pos1 = [int(x) for x in bottom_left]
        pos2 = [int(x) for x in top_left]
        pos3 = [int(x) for x in top_right]
        pos4 = [int(x) for x in bottom_right]

        pygame.draw.line(screen,self.BLACK,pos1,pos2,2) #left side
        pygame.draw.line(screen,self.BLACK,pos2,pos3,2) #top side
        pygame.draw.line(screen,self.BLACK,pos3,pos4,2) #right side
        pygame.draw.line(screen,self.BLACK,pos4,pos1,2) #bottom side

            
    def draw_barrier(self,screen):
        """
        Function to draw barrier

        Parameters:
            - idx (int): index to an array of buffer
        """

        for coor_start, coor_end in self.barrier_pos_pair:

            coor_start_py = self.trans_World2pygame(coor_start[0],coor_start[1])
            coor_end_py = self.trans_World2pygame(coor_end[0],coor_end[1])

            pos1 =  [int(x) for x in coor_start_py]
            pos2 =  [int(x) for x in coor_end_py]

            pygame.draw.line(screen,self.RED,pos1,pos2,4)


    def draw_rays(self,idx,screen):
        """
        Function to draw rays of LIDAR

        Parameters:
            - idx (int): index to an array of buffer
        """
        i = 0
        data_length = int(self.num_lidarRays/self.lidar_batch_num)

        lidar_pos_pair = self.lidar_pos_pair_buffer[idx]
        for coor_start, coor_end in lidar_pos_pair:

            coor_start_py = self.trans_World2pygame(coor_start[0],coor_start[1])
            coor_end_py = self.trans_World2pygame(coor_end[0],coor_end[1])

            pos1 =  [int(x) for x in coor_start_py]
            pos2 =  [int(x) for x in coor_end_py]

            
            pygame.draw.line(screen,self.YELLOW,pos1,pos2,1)

    def divided_lidar_batch(self):
        
        nn_data = []
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
            nn_data.append(min(batch))

        return nn_data

    def calculate_reward(self,prev_y):
        """
        Function to calculate reward for each step
        
        Returns:
            - rewards (float): reward of the step
        """
        min_batch_read = self.divided_lidar_batch()
        proximity_penalty = 0
        ray_length = 1.5 * self.robot_length_vir
        for read in min_batch_read:
            proximity_penalty += ((ray_length - read)/ray_length)*(0.1/self.lidar_batch_num)
        
        reward = 0.6*((prev_y+self.dy_read*self.dt*self.step_num)/(self.screen_height*(1/self.alpha))) - proximity_penalty
        
        time_penalty = 0.1*(self.time_passed/self.max_time)

        if self.reach_finish:
            reward = reward + 0.4 - time_penalty

        elif self.collide:
            reward = -1

        elif self.time_out:
            reward = reward - time_penalty

        
        return reward


    def reset_env(self):
        """
        function to reset environment

        Returns:
            - returned_obs (list of float): current observation of the robot
        """
        
        self.my_robot.reset()
        #positon and state of robot based on the reading from the robot model
        self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read = self.my_robot.get_state()

        #position of the robot in real and virtual based on our coordinate taken from the middle of the robot
        self.x_real = self.arena_width/2
        self.y_real = self.robot_length_real/2 + 0.05

        self.x_vir = self.x_real*self.alpha
        self.y_vir = self.y_real*self.alpha

        self.time_passed = 0

        #number of lidar rays place to save lidar reading
        self.lidar_dis_read = [] #real world reading
        self.lidar_dis_vir = [] #vir world reading

        #buffer for replay in pygame
        self.lidar_pos_pair_buffer = []
        self.robot_pos_buffer = []
        self.wheelL_pos_buffer = []
        self.wheelR_pos_buffer = []

        self.barrier_pos_pair = []

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
        self.randomize_barrier_pos()
        self.calc_rays_pos()
        self.calc_lidar_distance()

        returned_obs = [self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read,self.time_passed]
        
       
        
        return [returned_obs,self.lidar_dis_read]


    def step(self,wL, wR):
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
            while i<self.step_num:
                self.my_robot.computeKinematic(wR,wL)
                self.time_passed = self.time_passed + self.dt
                prev_y = self.y_real
                #positon and state of robot based on the reading from the robot model
                self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read = self.my_robot.get_state()

                #position of the robot in real and virtual based on our coordinate taken from the middle of the robot
                self.x_real = self.x_real + self.dx_read*self.dt
                self.y_real = self.y_real + self.dy_read*self.dt

                self.x_vir = self.x_real*self.alpha
                self.y_vir = self.y_real*self.alpha
                
                #calculate new pos
                self.calc_robot_pos()
                self.calc_wheel_pos()
                self.calc_rays_pos()

                #check distance of LIDAR,collision, and whether the robot already touch finsih line
                self.calc_lidar_distance()
                self.robot_collision_check()
                self.wheel_collision_check()
                self.check_timeout()
                self.check_finsih()


                if self.collide or self.reach_finish or self.time_out:
                    break
                
                i += 1

            reward = self.calculate_reward(prev_y)
            
            if self.collide or self.reach_finish or self.time_out:
                self.episode_end = True

            returned_obs = [self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read,self.time_passed]
            
            returned_values = [returned_obs,self.lidar_dis_read,reward,self.episode_end]

            return returned_values
        
        else:
            reward = 0.0
            returned_obs = [self.x_read,self.y_read,self.theta_read,self.dx_read,self.dy_read,self.dTheta_read,self.time_passed]
            
            
            returned_values = [returned_obs,self.lidar_dis_read,reward,self.episode_end]

            return returned_values


    def test_arena(self):
            """Main loop to render the drawing without running simulation."""
            screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            running = True
            while running:
                screen.fill(self.WHITE)  # Clear screen
                
                self.draw_robot(-1,screen)  # Draw the robot
                self.draw_wheel(-1, screen)  # Draw the wheels
                self.draw_barrier(screen)  # Draw barriers
                self.draw_rays(-1,screen)  # Draw LIDAR rays

                pygame.display.flip()  # Update screen

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
            
            pygame.quit()         


        
    def run_replay(self):
        loop = 0
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        clock = pygame.time.Clock()
        while loop < len(self.robot_pos_buffer):
            
            screen.fill(self.WHITE)  # Clear screen
                
            self.draw_robot(loop,screen)  # Draw the robot
            self.draw_wheel(loop,screen)  # Draw the wheels
            self.draw_barrier(screen)  # Draw barriers
            self.draw_rays(loop,screen)  # Draw LIDAR rays

            pygame.display.flip()  # Update screen
            clock.tick(60)
            loop=loop+1

        while running:
            screen.fill(self.WHITE)  # Clear screen
            
            self.draw_robot(-1,screen)  # Draw the robot
            self.draw_wheel(-1, screen)  # Draw the wheels
            self.draw_barrier(screen)  # Draw barriers
            self.draw_rays(-1,screen)  # Draw LIDAR rays

            pygame.display.flip()  # Update screen

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        
        pygame.quit()
    





        


    
