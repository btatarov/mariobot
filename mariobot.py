import retro
import numpy as np
import cv2
import neat
import random
import gzip
import pickle
import os

main_threads = 12
env = None
bot = None

def eval_genome(genome, config):
    global env

    if env is None:
        rnd = random.getrandbits(128)
        record_dir = './records/' + str(rnd)
        os.mkdir(record_dir)
        env = retro.make(game='Mario-Nes', state='Level1-NoBg', record=record_dir)
        env.reset()
        env.render()

    fitness = bot.eval_genome(genome, config, env)
    return fitness


class MarioBot(object):
    def __init__(self, threads=12, generation=0):
        self.threads = threads
        self.generation = generation

    def run(self):
        if self.generation > 0:
            with gzip.open('checkpoints/neat-checkpoint-{0}'.format(self.generation)) as f:
                self.generation, self.config, population, species_set, rndstate = pickle.load(f)
                self.generation += 1
                random.setstate(rndstate)
                self.population = neat.Population(self.config, (population, species_set, self.generation))
        else:
            self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 'config')

            self.population = neat.Population(self.config)

        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        self.population.add_reporter(neat.Checkpointer(generation_interval=200, filename_prefix='checkpoints/neat-checkpoint-'))

        self.evaluator_thread = neat.ParallelEvaluator(self.threads, eval_genome, 4000)
        winner = self.population.run(self.evaluator_thread.evaluate, 200) # 200 generations

        with open('checkpoints/winner', 'wb') as output:
            pickle.dump(winner, output, 1)

    def eval_genome(self, genome, config, env):
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape
        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        fitness_current = 0
        fitness_max = 0
        cur_reward = 0
        x_max = 0

        frame = 0
        frame_skip = 8
        counter = 0
        done = False

        while not done:
            frame += 1
            if frame % frame_skip == 0:
                env.render()

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))[3:,]
            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)
            ob, rew, done, info = env.step(nnOutput)

            # powerup on screen
            if info['powerup'] == 2:
                cur_reward += 100

            # killing an enemy
            if info['enemy-state'] != 0:
                cur_reward += 200

            # if jumping to the right
            if info['jump'] > 0 and x_max < (info['x-screen'] * 255 + info['x']):
                cur_reward += 4

            # hero moving to the left at the beginning
            if (info['x-screen'] * 255 + info['x']) < 40:
                cur_reward = cur_reward - 1

            # pitfalls
            if (info['y'] * info['y-screen']) > 176:
                cur_reward -= 100
                done = True

            # hero dies
            if info['lives'] < 2:
                cur_reward -= 100
                done = True

            if x_max < (info['x-screen'] * 255 + info['x']):
                x_max = info['x-screen'] * 255 + info['x']

            fitness_current = (x_max - 40) * 3
            fitness_current += cur_reward
            fitness_current += (info['score'] * 2)

            if fitness_current > fitness_max:
                fitness_max = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print('Genome:', genome.key, '\tFit:', fitness_current)

            genome.fitness = fitness_current

        return genome.fitness


if __name__ == '__main__':
    bot = MarioBot(threads=main_threads, generation=0)
    bot.run()
