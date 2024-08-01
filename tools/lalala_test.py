import torch
import torchvision
from torchvision import transforms
from PIL import Image

 


def main():
    test_path = "/home/lala/openpcdet/OpenPCDet/data/kitti/testing/image_2/000001.png"
    test_img = Image.open(test_path)
    #test_img.show()
    tensor_trans = transforms.ToTensor()
    tensor_PIL_img = tensor_trans(test_img)
    print(type(tensor_PIL_img))
    print(tensor_PIL_img.shape)
    model = torch.load("/home/lala/openpcdet/OpenPCDet/output/cfgs/kitti_models/pointpillar/default/ckpt/checkpoint_epoch_100.pth")
    print(model.keys())  
    
    # device = torch.device("cpu")
    # model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(tensor_PIL_img)
    print(output)

if __name__ == '__main__':
    main()
