import retro
import numpy as np
import cv2
import neat
import random
import gzip
import pickle
import os
import uuid

main_threads = 12
bot = None
env = None
env_id = None


def eval_genome(genome, config):
    global env, env_id

    if env is None:
        env_id = str(uuid.uuid4())
        record_dir = './records/' + env_id
        os.mkdir(record_dir)
        env = retro.make(game='Mario-Nes', state='Level1-NoBg', record=record_dir)

    fitness = bot.eval_genome(genome, config, env, env_id)
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

        stats = neat.StatisticsReporter()
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(stats)
        self.population.add_reporter(neat.Checkpointer(generation_interval=200, filename_prefix='checkpoints/neat-checkpoint-'))

        self.evaluator_thread = neat.ParallelEvaluator(self.threads, eval_genome, 4000)
        winner = self.population.run(self.evaluator_thread.evaluate, 200) # 200 generations

        with open('checkpoints/winner', 'wb') as output:
            pickle.dump(winner, output, 1)

    def eval_genome(self, genome, config, env, env_id):
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        screen = env.reset()
        screen_w, screen_h = int(env.observation_space.shape[1]/8), int(env.observation_space.shape[0]/8)
        movie_id = '%06d' % (env.movie_id - 1)

        x_max = 0
        reward = 0
        fitness_current = 0
        fitness_max = 0

        frame = 1
        frame_skip = 8
        counter = 0
        done = False

        while not done:
            if frame % frame_skip == 0:
                env.render()
                frame = 1
            else:
                frame += 1

            screen = cv2.resize(screen, (screen_h, screen_w))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = np.reshape(screen, (screen_h, screen_w))[3:,]  # truncate HUD lines
            screen_array = np.ndarray.flatten(screen)

            net_output = net.activate(screen_array)
            screen, _, _, info = env.step(net_output)

            # powerup on screen
            if info['powerup'] == 2:
                reward += 100

            # killing an enemy
            if info['enemy-state'] != 0:
                reward += 200

            # if jumping to the right
            if info['jump'] > 0 and x_max < (info['x-screen'] * 255 + info['x']):
                reward += 4

            # hero moving to the left at the beginning
            if (info['x-screen'] * 255 + info['x']) < 40:
                reward = reward - 1

            # pitfalls
            if (info['y'] * info['y-screen']) > 176:
                reward -= 100
                done = True

            # hero dies
            if info['lives'] < 2:
                reward -= 100
                done = True

            if x_max < (info['x-screen'] * 255 + info['x']):
                x_max = info['x-screen'] * 255 + info['x']

            fitness_current = (x_max - 40) * 3
            fitness_current += reward
            fitness_current += (info['score'] * 2)

            if fitness_current > fitness_max:
                fitness_max = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print('Genome: %-5d Fit: %-6d Movie: %s/%s' % (genome.key, fitness_current, env_id, movie_id))

            genome.fitness = fitness_current

        return genome.fitness


if __name__ == '__main__':
    bot = MarioBot(threads=main_threads, generation=0)
    bot.run()
