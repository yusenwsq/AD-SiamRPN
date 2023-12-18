# [AD-SiamRPN](https://www.mdpi.com/2072-4292/15/7/1731)

## 1. Environment setup
This code has been tested on Ubuntu 16.04, Python 3.6, Pytorch 0.4.1/1.2.0, CUDA 9.0.
Please install related libraries before running this code: 

```bash
pip install -r requirements.txt
```

## 2. Test
<table>
    <tr>
        <td colspan="2" align=center> Dataset</td>
        <td align=center>ADSiamRPN</td>
        <td align=center>BAENet</td>
        <td align=center>MFIHVT</td>
        <td align=center>MHT</td>
        <td align=center>DeepHKCF</td>
        <td align=center>BS-SiamRPN</td>
        <td align=center>SiamRPN++</td>
        <td align=center>DaSiamRPN</td>
    </tr>
    <tr>
        <td rowspan="2" align=center>HOT2022</td>
        <td>Success</td>
        <td>57.5</td>
        <td>61.6</td>
        <td>60.1</td>
        <td>58.4</td>
        <td>38.5</td>
        <td>53.3</td>
        <td>52.9</td>
        <td>55.8</td>
    </tr>
    <tr>
        <td>Precision</td>
        <td>86.1</td>
        <td>87.6</td>
        <td>89.1</td>
        <td>87.6</td>
        <td>73.7</td>
        <td>84.5</td>
        <td>83.4</td>
        <td>83.1</td>
     </tr>    

Download the pretrained model:  
[model](https://pan.baidu.com/s/1I3Tgyp1PA9Y3YSZlyHwJJg?pwd=bm1e) code: bm1e  

and put them into `models` directory.

Download the test result: 

[hot2022_result](https://pan.baidu.com/s/18iQoqRzOBa7qxO1AZiVbIw?pwd=4taf) code: 4taf


## 3. Acknowledgement
The code is implemented based on [pysot](https://github.com/STVIR/pysot). We would like to express our sincere thanks to the contributors.


## 4. Cite
If you use ADSiamRPN in your work please cite our papers:
```
	@article{wang2023ad,
	  title={AD-SiamRPN: Anti-Deformation Object Tracking via an Improved Siamese Region Proposal Network on Hyperspectral Videos},
	  author={Wang, Shiqing and Qian, Kun and Shen, Jianlu and Ma, Hongyu and Chen, Peng},
	  journal={Remote Sensing},
	  volume={15},
	  number={7},
  pages={1731},
	  year={2023},
	  publisher={MDPI}
	}
```

```
   @inproceedings{wang2022bs,
     title={BS-SiamRPN: Hyperspectral video tracking based on band selection and the Siamese region proposal network},
     author={Wang, ShiQing and Qian, Kun and Chen, Peng},
     booktitle={2022 12th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS)},
     pages={1--8},
  year={2022},
     organization={IEEE}
   }
```
