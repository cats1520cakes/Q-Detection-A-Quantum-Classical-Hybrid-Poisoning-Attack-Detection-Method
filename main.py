import torch
from dc import DCM
from QDetection.util import *
from QDetection.QDetectionfunc import *

set_seed(0)
# methodname = targeted_label_filpping  narcissus  badnets
import torch
from torchvision import models, transforms



def exp(methodname, poi_rate, subset_num):
    # load dataset
    data_transform = transforms.Compose([transforms.ToTensor(),])
    train_path = 'D:\QuantumDeepLearningOfHHQ\IJCAI Qdetection Github Code\dataset\gtsrb_dataset.h5'
    # Load the poisoned train dataset
    trainset = h5_dataset(train_path, True, None)
    train_poi_set, poi_idx = poi_dataset(trainset, poi_methond=methodname, transform=data_transform, poi_rates=poi_rate/100, random_seed=0, tar_lab=38)
    testset = h5_dataset(train_path, train=False, transform=None)

    # random_indices = np.random.choice(len(trainset), subset_num, replace=False)
    # random_test_acc, random_target_acc = test_resnet18_accuracy(train_poi_set, testset, random_indices)
    # print("random_test_acc:",random_test_acc, "random_target_acc:",random_target_acc)

    # baseline_indices = create_baseline_dataset(trainset, num_samples=4000)
    # print('\nTesting model trained on baseline dataset (randomly selected 4000 samples):')
    # baseline_test_acc, baseline_target_acc = test_resnet18_accuracy(trainset, testset, baseline_indices)
    # print("baseline_test_acc:",baseline_test_acc, "baseline_target_acc:",baseline_target_acc)
    
    dc_idx = DCM(train_poi_set, subset_num)
    print('NCR for DCM is: %.3f%%' % get_NCR(train_poi_set, poi_idx, dc_idx))
    # print("model trained on DCM filtered dataset:",test_resnet18_accuracy(trainset,testset,dc_idx))


    class Args:
        # Number of classes in sifting dafast
        num_classes = 43
        # Number of sifter
        num_sifter = 4
        # Number of sifting epoch
        res_epochs = 1
        # Number of warm epoch before sifting start
        warmup_epochs = 1
        # Batch size in sifting
        batch_size = 44 # 179
        # Number of workers in dataloader
        num_workers = 8
        # Learning rate for vent
        v_lr = 0.0005
        # Virtual update learning rate
        meta_lr = 0.1
        # Top k will be select to compute gradient
        top_k = 15
        # Learning rate for gradinet selection model
        go_lr = 1e-1
        # Number of activation for gradinet selection model
        num_act = 4
        momentum = 0.9
        nesterov = True
        # Random seed
        random_seed = 0
        device ='cuda' if torch.cuda.is_available() else 'cpu'
    args=Args()

        
    Meta_Sift_idx = Meta_Sift(args, train_poi_set, total_pick=subset_num)
    print('NCR for Meta-Sift is: %.3f%%' % get_NCR(train_poi_set, poi_idx, Meta_Sift_idx))
    # print("model trained on Meta-Sift filtered dataset:", test_resnet18_accuracy(trainset, testset, Meta_Sift_idx))


    # We do not recommend running Q-Detection_IM because the CIM simulation speed is relatively slow, although the result will be better. It is recommended to run Q-Detection_QA

    # QDetectionIM_idx = QDetection_IM(args, train_poi_set, total_pick=subset_num)
    # # NCR for Q-Detection
    # print('NCR for Q-Detection_IM Sift is: %.3f%%' % get_NCR(train_poi_set, poi_idx, QDetectionIM_idx))
    # print("model trained on Q-Detection_IM filtered dataset:",test_resnet18_accuracy(trainset,testset,QDetectionIM_idx))
    
    QDetectionQA_idx = QDetection_QA(args, train_poi_set, total_pick=subset_num)
    print('NCR for Q-Detection_QA Sift is: %.3f%%' % get_NCR(train_poi_set, poi_idx, QDetectionQA_idx))
    # print("model trained on Q-Detection_QA filtered dataset:",test_resnet18_accuracy(trainset,testset,QDetectionQA_idx))
    


if __name__ == '__main__':

    subset_num = 4000
    methodname = "targeted_label_filpping"
    for rate in [ 3, 5, 10, 20, 30]: #
        print("targeted_label_filpping  : ", rate)
        exp(methodname, rate, subset_num)

    # methodname = "narcissus"
    # for rate in [ 30 ]:  #3, 5 ,10 , 20
    #     print("narcissus  : ", rate)
    #     exp(methodname, rate, subset_num)
    #
    # methodname = "badnets"
    # for rate in [ 30 ]:  #  1, 3, 5 ,8, 10, 20
    #     print("badnets  : ", rate)
    #     exp(methodname, rate, subset_num)

    #
    # poi_rate = 16.67
    # methodname = "targeted_label_filpping"
    # for subset_num in [26000]:
    #     print("targeted_label_filpping  : ", subset_num)
    #     exp(methodname, poi_rate, subset_num)

    # poi_rate = 10
    # methodname = "narcissus"
    # for subset_num in [2000, 4000, 8000, 20000, 26000]:
    #     print("narcissus  : ", subset_num)
    #     exp(methodname, poi_rate, subset_num)

    # poi_rate = 33
    # methodname = "badnets"
    # for subset_num in [ 26000]:
    #     print("badnets  : ", subset_num)
    #     exp(methodname, poi_rate, subset_num)