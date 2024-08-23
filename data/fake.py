import os
dir = "/home/ubuntu/dmj/DeepfakeImageDetection/checkpoints/NPR_effnetb12024_08_21_15_18_02/prediction11.csv"
fake_dir = "/home/ubuntu/dmj/DeepfakeImageDetection/data/fake.csv"

def replace(dir, fake_dir):
    with open(fake_dir, 'r') as f:
        fake = f.readlines()
    fake = [ x.split(",")[0] for x in fake ]

    with open(dir, 'r') as f:
        data = f.readlines()

    new_dir = os.path.join(os.path.split(dir)[0], "prediction_replacefake.csv")
    with open(new_dir, 'w') as f:
        for line in data:
            if line.split(",")[0] in fake:
                f.write(  line.split(",")[0] + ",1.0\n"  )
            else:
                f.write(line)

if __name__ == "__main__":
    replace(dir, fake_dir)