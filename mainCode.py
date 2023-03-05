import numpy as np
import cv2
import random
import math

pixPerUnit = 50
imgSize = 50
N = 1000
circleWeight = 40
filterSpeed = 1.5 #.5 is slow, 3 is fast (too fast and the particles may lose track of the drone, too slow and the particles may not converge on the drone)
droneSpeed = 2
 #drone speed in units (50 pixels = 1 unit) per time step
resamplingNoise = 0.1 #variance of resampling distribution
droneMovementNoise = .1 #variance of drone movement distribution
maxTimeSteps = 50
distanceThresh = 75
clusterSizeThresh = 900
clusterFound = False
numRuns = 1
displayMap = True

bayMap = cv2.imread('BayMap.png')
cityMap = cv2.imread('CityMap.png')
marioMap = cv2.imread('MarioMap.png')
fixedMap = bayMap.copy() # CHANGE ME TO CHANGE THE MAP

map = fixedMap.copy()
w = map.shape[1]
h = map.shape[0]
print(f"Width: {w}, Height: {h}")


class Drone ():
    def __init__(self): #initialize drone at random location, but not too close to the edges
        print(f"Map.shape: {map.shape}")
        self.x = round(random.uniform(imgSize/2, map.shape[1]-imgSize/2))
        self.y = round(random.uniform(imgSize/2, map.shape[0]-imgSize/2))

    def rand_move(self): #move drone in a random direction, but not too close to the edges
        validMove = False
        sz = round(imgSize/2)
        while(not validMove):
            theta = random.uniform(0, 2*math.pi)
            self.dX = droneSpeed*pixPerUnit*math.cos(theta)
            self.dY = droneSpeed*pixPerUnit*math.sin(theta)
            totalDX = self.dX + random.gauss(0, droneMovementNoise*pixPerUnit)
            totalDY = self.dY + random.gauss(0, droneMovementNoise*pixPerUnit)
            if(self.x + totalDX < w-sz and self.x + totalDX > sz and self.y + totalDY < h-sz and self.y + totalDY > sz):
                self.x += totalDX
                self.y += totalDY
                validMove = True
        print(f"Drone Position: ({self.x}, {self.y}")
     
    def draw_drone (self): #draw drone on map
        x = round(self.x)
        y = round(self.y)
        sz = round(imgSize/2)
        cv2.rectangle(map, (self.corners[0], self.corners[1]), (self.corners[2], self.corners[3]), (255, 0, 0), 2)
        cv2.line(map, (x, 0), (x, y-sz), (255, 0, 0), 2)
        cv2.line(map, (x, y+sz), (x, h), (255, 0, 0), 2)
        cv2.line(map, (0, y), (x-sz, y), (255, 0, 0), 2)
        cv2.line(map, (x+sz, y), (w, y), (255, 0, 0), 2)
        cv2.circle(map, (x, y), 4, (255, 0, 0), 2)

    def get_pos(self): #return drone position
        return (round(self.x), round(self.y))
    
    def show_image(self): #return drone image
        # TopLeftX, TopLeftY, BottomRightX, BottomRightY
        self.corners = [round(self.x-imgSize/2), round(self.y-imgSize/2), round(self.x+imgSize/2), round(self.y+imgSize/2)]
        if self.corners[0] < 0: self.corners[0] = 0
        if self.corners[1] < 0: self.corners[1] = 0
        if self.corners[2] > w: self.corners[2] = w
        if self.corners[3] > h: self.corners[3] = h
        #print(f"corners: {self.corners}")
        return fixedMap[self.corners[1]:self.corners[3], self.corners[0]:self.corners[2]]


class Particles():
    def __init__(self):
        b = imgSize/2 #border around edge of map
        numPixWide = round(math.sqrt(1000*h/w))
        numPixHigh = round(1000/numPixWide)
        self.xy = np.mgrid[b:w-b:complex(0,numPixWide) , b:h-b:complex(0,numPixHigh)].reshape(2,-1).T.astype(int) #uniformly distribute particles, but not too close to the edges
        self.N = self.xy.shape[0]
        print(f"W: {numPixWide}, H: {numPixHigh}")
        print(f"Number of Particles: {self.N}")
        self.weights = np.full(self.N, 1/self.N)        
        self.draw_particles()


    def resample(self):    # Implement weighted importance sampling with replacement
        weightsNorm = self.weights / np.sum(self.weights)
        rng = np.random.default_rng()
        resampledParticles = rng.choice(self.xy, p=weightsNorm, size = (self.N,))
        for particle in resampledParticles:
            particle[0] += round(random.gauss(0, resamplingNoise*pixPerUnit))
            particle[1] += round(random.gauss(0, resamplingNoise*pixPerUnit))
        self.xy = resampledParticles


    def draw_particles(self): #update particle positions
        for i in range(self.N):
            cv2.circle(map, (round(self.xy[i][0]), round(self.xy[i][1])), round(self.weights[i]*circleWeight), (0, 0, 255), 2)
    
    def move_particles(self):
        self.xy[:,0] += round(drone.dX)
        self.xy[:,1] += round(drone.dY)

    def compare_imgs_mse(self, img1, img2): #compare drone image to reference image and return a measure of similarity
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        h = img1.shape[0]
        w = img1.shape[1]
        #print("Ref Image: ", img1.shape)
        #print("Drone Image: ", img2.shape)
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff**2)
        mse = err/(float(h*w))
        return mse
    
    def compare_imgs_hist(self, img1, img2):
        # Calculate the histograms
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0 ,256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0 ,256])
        # Compare the histograms using the correlation coefficient
        return(1- cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))

    def get_ref_image(self, i):
        sz = round(imgSize/2)
        #print(f"Ref Image: {round(self.xy[i][0]-sz)} to {round(self.xy[i][0]+sz)} and {round(self.xy[i][1]-sz)} to {round(self.xy[i][1]+sz)}")
        return(fixedMap[round(self.xy[i][1]-sz):round(self.xy[i][1]+sz), round(self.xy[i][0]-sz):round(self.xy[i][0]+sz)]) #get reference image from particle
    
    def get_weights(self):
        droneImg = drone.show_image()
        for i in range(self.N):
            self.weights[i] = self.compare_imgs_hist(self.get_ref_image(i), droneImg)**filterSpeed
        #print(f"weights: {self.weights}")
    
    def find_best_match(self):
        self.bestMatchIndex = np.argmax(self.weights)
        print(f"Best Match: {self.bestMatchIndex}  Location: {self.xy[self.bestMatchIndex]}")
        cv2.circle(map, (round(self.xy[self.bestMatchIndex][0]), round(particles.xy[self.bestMatchIndex][1])), 4, (0, 255, 255), 10)
    
    def distance(self, p1, p2): #calculate distance between two particles
        return (math.sqrt((self.xy[p1][0] - self.xy[p2][0])**2 + (self.xy[p1][1] - self.xy[p2][1])**2))
    
    def check_cluster(self): #check how many particles are near the best match
        clusterSize = 0
        for i in range(self.N):
            if self.distance(self.bestMatchIndex, i) < distanceThresh:
                clusterSize+=1
        print(f"Cluster Size: {clusterSize}")
        if clusterSize > clusterSizeThresh:
            print(f"\nCLUSTER FOUND!  Time Steps: {timeStep}\n")
            return True
        else:
            return False
        
    def check_error(self):
        return(math.sqrt((self.xy[self.bestMatchIndex][0] - drone.x)**2 + (self.xy[self.bestMatchIndex][1] - drone.y)**2))


def setup_loop():
    drone.show_image()
    drone.draw_drone()
    particles.draw_particles()
    if displayMap:
        cv2.imshow('MAP', map)
        cv2.imshow('Drone View', drone.show_image())
        cv2.waitKey(0)

def main_loop():
    global clusterFound
    drone.rand_move()
    particles.move_particles()
    particles.resample()
    particles.get_weights() 
    particles.draw_particles()
    drone.draw_drone()
    particles.find_best_match()
    if particles.check_cluster():
        clusterFound = True
    if displayMap:
        cv2.imshow('MAP', map)
        cv2.imshow('Drone View', drone.show_image())
        cv2.waitKey(0)

if __name__ == "__main__":
    localizationSpeed = np.empty(numRuns)
    accuracy = np.empty(numRuns)
    for i in range(numRuns):
        clusterFound = False
        drone = Drone()
        particles = Particles()
        setup_loop()
        for timeStep in range(maxTimeSteps):
            map = fixedMap.copy() #reset map
            main_loop()
            if clusterFound:
                print(f"ERROR: {particles.check_error()} pixels")
                break
        cv2.destroyAllWindows()
        localizationSpeed[i] = timeStep
        accuracy[i] = particles.check_error()
    print(f"Localization Speed: {np.mean(localizationSpeed)}")
    print(f"Accuracy: {np.mean(accuracy)}")




