import argparse
import os
import numpy as np



parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir", type=str, default="", help="folder of the results"
)
opt = parser.parse_args()
print(opt)


file_path = opt.dir

result_path = os.path.join(file_path,"results.npy")
data = np.load(result_path, allow_pickle=True)
data_dict = data.item()

num_samples = data_dict['num_samples']
num_repetitions = data_dict['num_repetitions']
all_text = data_dict['text']
all_lengths = data_dict['lengths']
all_motions = data_dict['motion']
batch_size = num_samples

sample_npy_template = 'sample{:02d}_rep{:02d}.npy'
sample_text_template = 'sample{:02d}.txt'

for sample_i in range(num_samples):
    for rep_i in range(num_repetitions):
        length = all_lengths[rep_i*batch_size + sample_i]
        motion = all_motions[rep_i*batch_size + sample_i].transpose(2, 0, 1)[:length]
        save_npy = sample_npy_template.format(sample_i, rep_i)
        np.save(os.path.join(file_path,save_npy),motion)
    caption = all_text[sample_i]
    save_text = sample_text_template.format(sample_i)
    with open(os.path.join(file_path,save_text), 'w') as fw:
        fw.write(caption)