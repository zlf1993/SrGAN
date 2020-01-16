from alfred.dl.tf.common import mute_tf
mute_tf()
from alfred.utils.mana import welcome
import os
from data import DIV2K
from model.srgan import generator, discriminator
from trainer import SrganTrainer, SrganGeneratorTrainer


welcome('https://gitlab.com/StrangeAI/super_resolution_tf2')
weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)
os.makedirs(weights_dir, exist_ok=True)


def train():
    div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
    div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')

    train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
    valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)
    
    pre_trainer = SrganGeneratorTrainer(model=generator(), checkpoint_dir='.ckpt/pre_generator')
    pre_trainer.train(train_ds,
                    valid_ds.take(10),
                    steps=1000000, 
                    evaluate_every=1000, 
                    save_best_only=False)
    pre_trainer.model.save_weights(weights_file('pre_generator.h5'))

    gan_generator = generator()
    gan_generator.load_weights(weights_file('pre_generator.h5'))

    gan_trainer = SrganTrainer(generator=gan_generator, discriminator=discriminator())
    gan_trainer.train(train_ds, steps=200000)

    gan_trainer.generator.save_weights(weights_file('gan_generator.h5'))
    gan_trainer.discriminator.save_weights(weights_file('gan_discriminator.h5'))


if __name__ == "__main__":
    train()
