# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:57:23 2020

@author: emet
"""


import retro
import numpy as np
import pickle
import cv2
import neat


# before you run this code, it is helpful to play this video as it speeds up the network training process
# https://www.youtube.com/watch?v=rJ-h_l1WfRM


#Sets up an enviroment. Gym tells my code where to look in memory to see the information it needs about the Sonic game.
env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1') 
env.reset()

xpos_end = 0 #we will use these later
imgarray = []

def eval_genomes(genomes, config):

    #for each genome we generate, create a recurrent neural network for that genome
    for genome_id, genome in genomes:
        ob = env.reset() #observation (an image)
        #ac = env.action_space.sample() #action (a random action)
        
        #lnx = the x
        #lny = the y
        #lnc = the colors
        #shape has to do with the pixels on the screen
        lnx, lny, lnc = env.observation_space.shape
        
        lnx = int(lnx/8) 
        lny = int(lny/8) 
        
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 80
        xpos_max = 80 #sonic starts at 80
        speed = 0
        
        done = False
        
        while not done:
            
            env.render()
            
            frame += 1
        
            #how to get fewer input variables 101
            ob = cv2.resize(ob, (lnx, lny)) #resize the screen (smaller)
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY) #make the screen greyscale
            ob = np.reshape(ob, (lnx, lny)) #make everything fit togethor nice
            
            #basically np.flatten()
            for x in ob:
                for y in x:
                    imgarray.append(y)
            
            nnOutput = net.activate(imgarray)
        
            #env.step(action) dumps a ton of useful information
            #ob is the image of the screen at the time of the action
            #rew is the amount of reward you earn from what was in the scenario file
            #done is whether sonic has lot all of his lives or reached the end of the level
            #info is a dictionary of all of the values that you have set in data
            ob, rew, done, info = env.step(nnOutput) #Take the nn output and use it to affect the Sonic enviroment
        
            imgarray.clear() #probs not nessisary
            
            xpos = info['x']
            xpos_end = info['screen_x_end'] #final x position in whatever level you load
            
            #every time Sonic goes further to the right than he has ever been he gets a treat
            if xpos > xpos_max:
                fitness_current += 1
                speed = xpos - xpos_max
                
                if speed > 3:
                    fitness_current += speed

                xpos_max = xpos
                

                
            #if Sonic gets to the end of the level we are done
            if xpos < xpos_end and xpos > (xpos_end - 500) and xpos > 500:
                fitness_current += 100000
                print("WE DID IT!")
                done = True
                
            #count how many times Sonic has failed to get rewarded
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            
            #if Sonic failes to get rewarded 250 times then we give up on him
            #done will also become true automatically if Sonic dies 3 times
            if done or counter == 300:
                done = True
                print(genome_id, fitness_current)
            
            #unitl Sonic fails to get rewarded or dies, keep modifying the fitness score of this genome
            genome.fitness = fitness_current
                
#reads the config file
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'C:\\Users\\emet\\Desktop\\COMP\\config-feedforward.txt')

#usues default pop number found in config file. Sets up a bunch of variables including genomes. 
p = neat.Population(config)
    
p.add_reporter(neat.StdOutReporter(True)) #adds a reporter that gives you updates in the console
stats = neat.StatisticsReporter() 
p.add_reporter(stats) #now it gives you stats too
p.add_reporter(neat.Checkpointer(10)) #saves your progress periodically

winner = p.run(eval_genomes) # train the network!

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)




            #Sonic go fast == good
            #if frame % 50 == 0: #mess with speeeed
                #speed = xpos - lastXPos
                #lastXPos = xpos
                #if speed > 0:
                    #fitness_current += speed
                    #if speed > 90: 
                        #print("Gotta go fast: " + str(speed))
    
   