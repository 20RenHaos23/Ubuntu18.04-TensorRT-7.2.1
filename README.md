# Ubuntu18.04-TensorRT-7.2.1
在Ubuntu18.04安裝TensorRT

目的
---
因為在NVIDIA Jetson Nano 2GB使用[NVIDIA-AI-IOT/trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)偵測人體時，每次執行都需等待十分鐘以上的時間，程式才會正式開始執行。造成每次修改程式後，皆須等待很久才能知道此次修改是否成功。所以將閒置電腦(裝上顯示卡)安裝也可以執行[NVIDIA-AI-IOT/trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose)的環境，並在此電腦進行程式的修改與測試，程式沒有問題後再轉移到Jetson Nano執行，省下等待的時間。

資訊
---
系統 : Ubuntu 18.04

顯示卡 : NVIDIA GeForce GTX 1650

安裝步驟
---
1. 
```
sudo apt updata
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/1.png)

2. 
```
sudo apt upgrade
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/2.png)

3. 安裝顯示卡驅動

參考[Ubuntu 20.04中安裝nvidia-driver-460版 & CUDA-11.4.2版 & cuDNN｜Install nvidia-driver-460 & CUDA-11.4.2 & cuDNN in Ubuntu 20.04](https://medium.com/@scofield44165/ubuntu-20-04%E4%B8%AD%E5%AE%89%E8%A3%9Dnvidia-driver-cuda-11-4-2%E7%89%88-cudnn-install-nvidia-driver-460-cuda-11-4-2-cudnn-6569ab816cc5)前面教學的步驟
```
sudo lshw -numeric -C display
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/3.png)


4. 
```
sudo apt-get purge nvidia*
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/4.png)

5. 
```
sudo add-apt-repository ppa:graphics-drivers
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/5.png)


6. 
```
sudo apt-get update
sudo apt upgrade
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/6.png)


7. 
```
ubuntu-drivers list
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/7.png)

8. 
```
sudo apt install nvidia-driver-470
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/8.png)
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/9.png)

9. 
```
reboot
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/10.png)

10. 
```
nvidia-smi
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/11.png)


---

11. 安裝CUDA 11.1

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/12.png)

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/13.png)

12. 
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/14.png)

13. 
```
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/15.png)

14. 
```
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/16.png)

15. 
```
sudo dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.0-455.23.05-1_amd64.deb
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/17.png)

16. 
```
sudo apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/18.png)

17. 
```
sudo apt-get update
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/19.png)


18. 
```
sudo apt-get -y install cuda
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/20.png)

19. 新增環境變數
```
vim ~/.bashrc
```
如果還沒有安裝vim，則先安裝vim
```
sudo apt install vim
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/21.png)

進入到指令模式 (command mode)之後，按「i」，進入「編輯模式 (insert mode)」。
在最底下新增下面兩行
```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
完成之後輸入「:wq」
```
source ~/.bashrc
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/22.png)

20. 
```
nvcc -V
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/23.png)


---

21. 安裝cuDnn 8.0.5

下載[cuDNN libcuDNN Library for Linux (x86_64)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.1_20201106/cudnn-11.1-linux-x64-v8.0.5.39.tgz)

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/24.png)

22. 在Downloads開啟Terminal
```
tar -xvf cudnn-11.1-linux-x64-v8.0.5.39.tgz
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/27.png)

23. 
```
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include #圖片上沒有在cudnn.h上面加到*
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64  
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn* 
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/28.png)

24. 測試cuDNN是否安裝成功

重新打開一個Terminal
```
git clone https://github.com/li-weihua/cudnn_samples_v8 
cd cudnn_samples_v8/mnistCUDNN
make clean && make
```
如果還沒有安裝git，則先安裝git
```
sudo apt install git
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/30.png)

如果遇到 fatal error: FreeImage.h: No such file or directory

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/31.png)

執行
```
sudo apt install libfreeimage3 libfreeimage-dev
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/32.png)

再執行一次
```
make clean && make
./mnistCUDNN
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/33.png)

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/34.png)

出現
```
Test passed!
```
代表成功


---

25. 安裝pip3
```
pip3 -V #應該是打-V或者--version，圖片上面打錯了
sudo apt install python-pip3
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/35.png)

26. 安裝pip
```
pip -V #應該是打-V或者--version，圖片上面打錯了
sudo apt install python-pip
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/36.png)


---

27. 安裝Pillow
```
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/37.png)



---

28. 安裝Pytorch 1.8.0
```
pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/38.png)

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/39.png)

29. 測試Pytorch是否安裝成功
```
python3
```

```
import torch
x = torch.rand(5, 3)
print(x)
torch.cuda.is_available()
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/40.png)


---

30. 安裝TensorRT 7.2.1

參考[TensorRT 介紹與安裝教學](https://medium.com/ching-i/tensorrt-%E4%BB%8B%E7%B4%B9%E8%88%87%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-45e44f73b25e)

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/41.png)

31. 
在Downloads開啟Terminal
```
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.1.6-ga-20201007_1–1_amd64.deb
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/42.png)

32. 
```
sudo apt-key add /var/nv-tensorrt-repo-cuda11.1-trt7.2.1.6-ga-20201007/7fa2af80.pub
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/43.png)

33. 
```
sudo apt-get update
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/44.png)

34. 
```
udo apt-get install tensorrt
```
![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/45.png)

35. 
```
sudo apt-get install python3-libnvinfer-dev
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/46.png)

36. 
```
dpkg -l | grep TensorRT
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/47.png)

37. 
```
dpkg-query -W tensorrt
```

![image](https://github.com/20RenHaos23/Ubuntu18.04-TensorRT-7.2.1/blob/main/README_img/48.png)

參考網址
---
[Ubuntu 20.04中安裝nvidia-driver-460版 & CUDA-11.4.2版 & cuDNN｜Install nvidia-driver-460 & CUDA-11.4.2 & cuDNN in Ubuntu 20.04](https://medium.com/@scofield44165/ubuntu-20-04%E4%B8%AD%E5%AE%89%E8%A3%9Dnvidia-driver-cuda-11-4-2%E7%89%88-cudnn-install-nvidia-driver-460-cuda-11-4-2-cudnn-6569ab816cc5)


[【vim #1】vim 的 新手/初學者 的基礎使用指令 與 個人常用功能總整理 (updated: 2022/12/11) - 嗡嗡的隨手筆記](https://www.wongwonggoods.com/all-posts/dev-tools/linux-editor/linux-vim/linux-ubuntu-vim/)


[fatal error: cudnn_version.h: No such file or directory · Issue #2356 · pjreddie/darknet](https://github.com/pjreddie/darknet/issues/2356)

[Installation - Pillow (PIL Fork) 10.0.1 documentation](https://pillow.readthedocs.io/en/stable/installation.html)

[Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

[TensorRT 介紹與安裝教學](https://medium.com/ching-i/tensorrt-%E4%BB%8B%E7%B4%B9%E8%88%87%E5%AE%89%E8%A3%9D%E6%95%99%E5%AD%B8-45e44f73b25e)

[Installation Guide :: NVIDIA Deep Learning TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian)

