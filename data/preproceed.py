import os

dir = '/mnt/share_data/dmj/phase1/'
target = '/mnt/share_data/dmj/phase1_converted/'

def convert_phase1_to_NPR(dir, target):
    files = os.listdir(dir)
    if "trainset_label.txt" not in files:
        print("Dir Error")
        return
    train_dic = {}

    with open(os.path.join(dir + 'trainset_label.txt'), 'r', encoding = 'utf8' ) as f:
        for line in f:
            file_label = line.strip().split(",")
            train_dic[file_label[0]] = file_label[1]

    train_dir = os.path.join(dir + 'trainset/')
    train_imgs = os.listdir(train_dir)
    length = str(len(train_imgs))
    for id,img in enumerate(train_imgs):
        # print('train:' + str(id) + '/' + length)
        if train_dic[img] == '0':
            os.system("cp %s %s"%(os.path.join(train_dir, img), os.path.join(target, 'train/0_real/' + img) ))
        else:
            os.system("cp %s %s"%(os.path.join(train_dir, img), os.path.join(target, 'train/1_fake/' + img) ))

    val_dic = {}
    with open(os.path.join(dir + 'valset_label.txt'), 'r', encoding = 'utf8' ) as f:
        for line in f:
            file_label = line.strip().split(",")
            val_dic[file_label[0]] = file_label[1]
    val_dir = os.path.join(dir + 'valset/')
    val_imgs = os.listdir(val_dir)
    length = str(len(val_imgs))
    for id,img in enumerate(val_imgs):
        # print('val:' + str(id) + '/' + length)
        if val_dic[img] == '0':
            os.system("cp %s %s"%(os.path.join(val_dir, img), os.path.join(target, 'val/0_real/' + img)  ))
        else:
            os.system("cp %s %s"%(os.path.join(val_dir, img), os.path.join(target, 'val/1_fake/' + img)  ))


if __name__ == '__main__':
    convert_phase1_to_NPR(dir,target)