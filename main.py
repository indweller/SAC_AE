import metaworld
import random

ml1 = metaworld.ML1('drawer-open-v2')
env = ml1.train_classes['drawer-open-v2']()
task = random.choice(ml1.train_tasks)
env.set_task(task)
obs = env.reset()
done = False

while not done:
    a = env.action_space.sample()
    obs, reward, done, info = env.step(a)